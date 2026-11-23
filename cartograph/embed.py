"""Optional semantic layer. Builds a brute-force vector index over all chunks (numpy memmap) using a
local sentence-transformers model — GPU-accelerated if torch sees CUDA, else CPU. Degrades to nothing
if `cartograph[semantic]` isn't installed, so the base product works without it.

Brute-force is fine to a few million chunks (~150ms/query). Past that, swap in FAISS — see docs."""
from __future__ import annotations


from .config import Config, index_dir
from .storage import Store

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
_MODEL = None


def available() -> bool:
    try:
        import numpy  # noqa
        import sentence_transformers  # noqa
        return True
    except Exception:
        return False


def _model(name: str):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        dev = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                dev = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                dev = "mps"                                   # Apple-silicon GPU
        except Exception:
            pass
        _MODEL = SentenceTransformer(name, trust_remote_code=True, device=dev)
        try:
            _MODEL.max_seq_length = 192  # speed; chunks are short
        except Exception:
            pass
    return _MODEL


def _load_previous(out, dim: int):
    """(chunk_ids, vectors memmap) of the last complete index, or (None, None)."""
    import numpy as np
    try:
        ids = np.load(out / "chunk_ids.npy")
        vecs = np.load(out / "vectors.npy", mmap_mode="r")
        if vecs.shape == (len(ids), dim):
            return ids, vecs
    except Exception:
        pass
    return None, None


def build_index(store: Store, cfg: Config, *, batch: int = 256, progress=None, full: bool = False) -> dict:
    """Embed chunks -> index/{vectors.npy, chunk_ids.npy, project_ids.npy}.

    Incremental by default: vectors of chunks already in the last complete index are copied, only new
    chunks are embedded (a changed file gets new chunk ids, so it is re-embedded; deleted chunks drop out).
    Text is fetched per batch instead of loading the whole corpus into memory. The new index is written
    to a temp folder and swapped in at the end, so an interrupted run leaves the previous index intact."""
    import shutil

    import numpy as np
    model = _model(cfg.embed_model)
    with store.cursor() as c:
        rows = c.execute("SELECT ch.id, f.project_id FROM chunks ch JOIN files f ON f.id=ch.file_id "
                         "ORDER BY ch.id").fetchall()
    if not rows:
        return {"vectors": 0}
    out = index_dir()
    out.mkdir(parents=True, exist_ok=True)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    pids = np.array([r[1] if r[1] is not None else -1 for r in rows], dtype=np.int64)
    dim = model.get_sentence_embedding_dimension()
    old_ids, old_vecs = (None, None) if full else _load_previous(out, dim)
    tmp = out.parent / (out.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    vecs = np.lib.format.open_memmap(tmp / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(ids), dim))
    todo = np.arange(len(ids))
    reused = 0
    if old_ids is not None and len(old_ids):
        order = np.argsort(old_ids)
        pos = np.searchsorted(old_ids[order], ids)
        pos = np.clip(pos, 0, len(old_ids) - 1)
        hit = old_ids[order][pos] == ids
        src = order[pos[hit]]
        dst = np.nonzero(hit)[0]
        for i in range(0, len(dst), 200_000):                 # copy in slices to bound memory
            vecs[dst[i:i + 200_000]] = old_vecs[src[i:i + 200_000]]
        reused = int(hit.sum())
        todo = np.nonzero(~hit)[0]
    for i in range(0, len(todo), batch):
        sel = todo[i:i + batch]
        want = [int(x) for x in ids[sel]]
        ph = ",".join("?" * len(want))
        with store.cursor() as c:
            text = {r[0]: r[1] for r in c.execute(f"SELECT id, chunk_text FROM chunks WHERE id IN ({ph})", want)}
        texts = [DOC_PREFIX + (text.get(k) or "")[:600] for k in want]
        emb = model.encode(texts, normalize_embeddings=True, batch_size=batch)
        vecs[sel] = emb.astype("float16")
        if progress and (i // batch) % 8 == 0:
            progress(f"embedded {i + len(sel)}/{len(todo)} new chunks ({reused} reused)")
    vecs.flush()
    del vecs
    np.save(tmp / "chunk_ids.npy", ids)
    np.save(tmp / "project_ids.npy", pids)
    old = out.parent / (out.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    out.rename(old)
    tmp.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    return {"vectors": len(ids), "embedded": int(len(todo)), "reused": reused, "dim": dim, "dir": str(out)}


class SemanticIndex:
    def __init__(self, cfg: Config) -> None:
        import numpy as np
        d = index_dir()
        self.vectors = np.load(d / "vectors.npy", mmap_mode="r")
        self.chunk_ids = np.load(d / "chunk_ids.npy")
        self.project_ids = np.load(d / "project_ids.npy")
        self.cfg = cfg

    @staticmethod
    def exists() -> bool:
        return (index_dir() / "vectors.npy").exists()

    def _ranked(self, query: str, fetch: int):
        import numpy as np
        qv = _model(self.cfg.embed_model).encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        sims = np.asarray(self.vectors @ qv.astype("float16"))
        fetch = min(fetch, sims.shape[0])
        idx = np.argpartition(-sims, fetch - 1)[:fetch]
        return idx[np.argsort(-sims[idx])]

    def search_chunk_ids(self, query: str, top_k: int = 30, fetch: int = 400) -> list[int]:
        try:
            idx = self._ranked(query, fetch)
        except Exception:
            return []
        return [int(self.chunk_ids[i]) for i in idx[:top_k]]
