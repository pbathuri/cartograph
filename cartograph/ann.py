"""Approximate nearest-neighbour search for big graphs — pure NumPy, no extra dependency.

Brute force over the vector memmap is exact but reads every vector per query: fine for a few hundred
thousand chunks, ~10 s per query at 6.5M. This layer is an IVF-PQ index:

  1. coarse k-means splits the vectors into `nlist` cells; a query visits the `nprobe` closest cells;
  2. inside them, 64-byte product-quantization codes of each vector's residual from its cell centre
     (stored contiguously per cell) give approximate scores for every candidate with table lookups;
  3. the best `rerank` candidates are re-scored exactly against the stored vectors.

So the final ranking is exact over a candidate set that almost always contains the true top hits.
NumPy-only on purpose: FAISS and PyTorch ship different OpenMP runtimes that abort the process when
both load (macOS), and agents call retrieval from processes that already run the embedding model.

It indexes only rows that are already embedded, so it works on a partial index and is refreshed as a
build proceeds."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

ANN_DIR = "ann"
META_FILE = "ann.json"
MIN_ROWS = 100_000           # below this brute force is already fast
_BLOCK = 8192


def available() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def _params(n: int, dim: int) -> tuple[int, int]:
    nlist = 1 << int(round(math.log2(max(16, min(8192, 4 * math.sqrt(n))))))
    m = next(k for k in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1) if dim % k == 0)
    return nlist, m


def meta(d: Path) -> dict:
    try:
        return json.loads((d / META_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _assign(X, C, spherical: bool, cnorm=None):
    """Nearest centroid per row, in blocks so the score matrix stays small."""
    import numpy as np
    out = np.empty(len(X), dtype=np.int32)
    half = None if spherical else 0.5 * (cnorm if cnorm is not None else (C * C).sum(1))
    for i in range(0, len(X), _BLOCK):
        s = X[i:i + _BLOCK] @ C.T
        if half is not None:
            s -= half
        out[i:i + _BLOCK] = s.argmax(1)
    return out


def _kmeans(X, k: int, iters: int, rng, spherical: bool):
    import numpy as np
    k = min(k, len(X))
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = _assign(X, C, spherical)
        counts = np.bincount(a, minlength=k)
        order = np.argsort(a, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        nz = counts > 0
        sums = np.zeros_like(C)
        sums[nz] = np.add.reduceat(X[order], starts[nz], axis=0)
        C[nz] = sums[nz] / counts[nz, None]
        empty = np.nonzero(~nz)[0]
        if len(empty):
            C[empty] = X[rng.choice(len(X), len(empty), replace=False)]
        if spherical:
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    return C


def build(d: Path, rows=None, *, min_rows: int = MIN_ROWS, progress=None, seed: int = 0,
          iters: int = 8, sample_size: int = 150_000) -> dict:
    """(Re)build the ANN index for the vectors in `d` over `rows` (default: every filled row)."""
    import numpy as np

    from .embed import _filled_rows
    vecs = np.load(d / "vectors.npy", mmap_mode="r")
    if rows is None:
        rows = np.nonzero(_filled_rows(vecs))[0]
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    n, dim = len(rows), vecs.shape[1]
    if n < min_rows:
        return {"skipped": f"only {n:,} embedded rows (ANN starts at {min_rows:,})"}
    t0 = time.time()
    nlist, m = _params(n, dim)
    dsub = dim // m
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(rows, size=min(n, sample_size), replace=False))
    S = np.asarray(vecs[sample], dtype=np.float32)
    if progress:
        progress(f"ANN: clustering {len(S):,} sample vectors into {nlist} cells")
    C = _kmeans(S, nlist, iters, rng, spherical=True)
    if progress:
        progress(f"ANN: training {m} product quantizers")
    R = S - C[_assign(S, C, spherical=True)]                    # PQ encodes the residual from the cell centre
    books = np.stack([_kmeans(np.ascontiguousarray(R[:, j * dsub:(j + 1) * dsub]), 256, iters, rng, spherical=False)
                      for j in range(m)])                       # (m, 256, dsub)
    del R
    bnorm = (books * books).sum(2)
    cell = np.empty(n, dtype=np.int32)
    codes = np.empty((n, m), dtype=np.uint8)
    step = 100_000
    for i in range(0, n, step):
        X = np.asarray(vecs[rows[i:i + step]], dtype=np.float32)
        cell[i:i + step] = _assign(X, C, spherical=True)
        X -= C[cell[i:i + step]]
        for j in range(m):
            codes[i:i + step, j] = _assign(np.ascontiguousarray(X[:, j * dsub:(j + 1) * dsub]), books[j],
                                           spherical=False, cnorm=bnorm[j])
        if progress:
            progress(f"ANN: encoded {min(i + step, n):,}/{n:,}")
    order = np.argsort(cell, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(cell, minlength=nlist))]).astype(np.int64)
    tmp = d / (ANN_DIR + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    np.save(tmp / "centroids.npy", C)
    np.save(tmp / "codebooks.npy", books)
    np.save(tmp / "rows.npy", rows[order])
    np.save(tmp / "codes.npy", codes[order])
    np.save(tmp / "offsets.npy", offsets)
    final = d / ANN_DIR
    old = d / (ANN_DIR + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    info = {"rows": int(n), "total": int(vecs.shape[0]), "dim": int(dim), "nlist": nlist, "m": m,
            "built_at": time.time(), "seconds": round(time.time() - t0, 1)}
    (d / META_FILE).write_text(json.dumps(info))
    return info


class Ann:
    """A loaded ANN index; `candidates()` returns row positions to re-score exactly."""

    def __init__(self, d: Path, nprobe: int = 64) -> None:
        import numpy as np
        a = d / ANN_DIR
        self.centroids = np.load(a / "centroids.npy")
        self.books = np.load(a / "codebooks.npy")
        self.rows = np.load(a / "rows.npy", mmap_mode="r")
        self.codes = np.load(a / "codes.npy", mmap_mode="r")
        self.offsets = np.load(a / "offsets.npy")
        self.nprobe = min(nprobe, len(self.centroids))
        self.meta = meta(d)

    @staticmethod
    def usable(d: Path, total_rows: int) -> bool:
        return (d / ANN_DIR / "codes.npy").exists() and meta(d).get("total") == total_rows

    def candidates(self, qv, k: int):
        import numpy as np
        q = np.asarray(qv, dtype=np.float32)
        cscore = self.centroids @ q
        cells = np.argpartition(-cscore, self.nprobe - 1)[:self.nprobe]
        cells.sort()
        spans = [(c, self.offsets[c], self.offsets[c + 1]) for c in cells if self.offsets[c + 1] > self.offsets[c]]
        if not spans:
            return np.empty(0, dtype=np.int64)
        codes = np.concatenate([self.codes[a:b] for _, a, b in spans])
        rows = np.concatenate([self.rows[a:b] for _, a, b in spans])
        m, _, dsub = self.books.shape
        table = np.einsum("jcd,jd->jc", self.books, q.reshape(m, dsub))      # (m, 256)
        approx = np.repeat(cscore[[c for c, _, _ in spans]], [b - a for _, a, b in spans]).astype(np.float32)
        for j in range(m):
            approx += table[j][codes[:, j]]
        k = min(int(k), len(approx))
        top = np.argpartition(-approx, k - 1)[:k]
        return np.asarray(rows[top], dtype=np.int64)
