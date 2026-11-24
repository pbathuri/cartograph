"""Optional semantic layer. Builds a brute-force vector index over all chunks (numpy memmap) using a
local sentence-transformers model — GPU-accelerated if torch sees CUDA or Apple MPS, else CPU. Degrades
to nothing if `cartograph[semantic]` isn't installed, so the base product works without it.

Builds are resumable: a killed `carto index` picks up from the rows it already wrote. The user's first
configured roots are embedded first, and a build in progress is already searchable.

Brute force is exact and fine to a few hundred thousand chunks; past ~100k embedded rows an IVF-PQ layer
(ann.py, NumPy-only) keeps queries in the tens of milliseconds."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import Config, db_path, index_dir
from .storage import Store

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
EMBED_CHARS = int(os.environ.get("CARTOGRAPH_EMBED_CHARS", "600"))
_MODEL = None
_FILES = ("vectors.npy", "chunk_ids.npy", "project_ids.npy")


def available() -> bool:
    """Whether the semantic extra is installed, without importing it (importing torch takes seconds)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("numpy", "sentence_transformers"))
    except (ImportError, ValueError):
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
        if dev != "cpu":
            try:
                _MODEL.half()             # vectors are stored as float16 anyway; ~15% faster on MPS
            except Exception:
                pass
    return _MODEL


def _dim(model) -> int:
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(fn())


def _default_batch(model) -> int:
    dev = str(getattr(model, "device", "cpu"))
    return 256 if dev.startswith("cuda") else 128 if dev.startswith("mps") else 64


def building_dir() -> Path:
    out = index_dir()
    return out.parent / (out.name + ".building")


def _carried_dir() -> Path:
    out = index_dir()
    return out.parent / (out.name + ".carried")


def _lock_path() -> Path:
    out = index_dir()
    return out.parent / (out.name + ".lock")


class BuildRunning(RuntimeError):
    """Another live process is building the index; a second builder would destroy its work."""

    def __init__(self, pid: int) -> None:
        super().__init__(f"an index build is already running (pid {pid})")
        self.pid = pid


def _acquire_lock() -> Path:
    """Claim the single-builder lock, or raise BuildRunning. A lock left by a dead process is taken over."""
    p = _lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(p.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and pid != os.getpid() and pid_alive(pid):
                raise BuildRunning(pid) from None
            p.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return p
    raise BuildRunning(0)


def _release_lock(p: Path) -> None:
    try:
        if int(p.read_text().strip() or 0) == os.getpid():
            p.unlink()
    except (OSError, ValueError):
        pass


def _complete(d: Path) -> bool:
    return all((d / n).exists() for n in _FILES)


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


def _build_started(tmp: Path) -> float:
    st = (tmp / "vectors.npy").stat()
    return float(getattr(st, "st_birthtime", 0) or st.st_mtime)


def _graph_mtime() -> float:
    p = db_path()
    times = [q.stat().st_mtime for q in (p, p.with_name(p.name + "-wal")) if q.exists()]
    return max(times) if times else 0.0


def _filled_rows(vecs, step: int = 200_000):
    """Rows a previous run finished writing. Unit vectors are never all-zero, so a row with both its
    first and last dims non-zero is complete; a row cut off mid-write is redone."""
    import numpy as np
    n, d = vecs.shape
    k = min(8, d)
    done = np.zeros(n, dtype=bool)
    for i in range(0, n, step):
        blk = np.asarray(vecs[i:i + step])
        done[i:i + step] = np.any(blk[:, :k] != 0, axis=1) & np.any(blk[:, -k:] != 0, axis=1)
    return done


def _resume(tmp: Path, ids, dim: int):
    """(vectors memmap r+, done mask) for a partial build of exactly these chunks, else None.

    A partial build records its chunk ids. One written by an older version has no ids file; it is
    trusted only when the graph has not been written since the build began (the ids can't have moved)."""
    import numpy as np
    try:
        vecs = np.load(tmp / "vectors.npy", mmap_mode="r+")
    except Exception:
        return None
    if vecs.shape != (len(ids), dim) or vecs.dtype != np.float16:
        return None
    if (tmp / "chunk_ids.npy").exists():
        try:
            if not np.array_equal(np.load(tmp / "chunk_ids.npy"), ids):
                return None
        except Exception:
            return None
    elif _graph_mtime() >= _build_started(tmp):
        return None
    return vecs, _filled_rows(vecs)


def _priority(store: Store, cfg: Config, pids):
    """Positions in embedding order: projects under the first configured root first, then the next
    root, and so on; within a root, smaller projects first so many projects become searchable early.
    Projects with no folder (live activity) lead; projects outside every root go last."""
    import numpy as np
    roots = [str(Path(r).expanduser()) for r in (cfg.roots or [])]
    with store.cursor() as c:
        paths = {int(r[0]): r[1] for r in c.execute("SELECT id, path FROM projects").fetchall()}

    def rank(pid: int) -> int:
        path = paths.get(pid)
        if not path:
            return -1
        for i, r in enumerate(roots):
            if Store._under(path, r):
                return i
        return len(roots)

    uniq, inv = np.unique(pids, return_inverse=True)
    ranks = np.array([rank(int(p)) for p in uniq], dtype=np.int64)[inv]
    sizes = np.bincount(inv)[inv]
    return np.lexsort((np.arange(len(pids)), sizes, ranks))


def _write_progress(tmp: Path, **kw) -> None:
    p = tmp / "progress.json"
    t = p.with_suffix(".json.tmp")
    t.write_text(json.dumps(kw))
    os.replace(t, p)


def pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":                     # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k32 = ctypes.windll.kernel32        # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)              # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259                # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def index_status() -> dict:
    """What the semantic index is doing: building (with counts/rate/ETA), complete, or absent."""
    tmp = building_dir()
    p = tmp / "progress.json"
    if p.exists():
        try:
            st = json.loads(p.read_text())
        except Exception:
            st = {}
        st.update(state="building", running=pid_alive(st.get("pid")))
        return st
    if _complete(index_dir()):
        try:
            import numpy as np
            n = int(np.load(index_dir() / "chunk_ids.npy", mmap_mode="r").shape[0])
        except Exception:
            n = 0
        return {"state": "complete", "done": n, "total": n, "running": False}
    return {"state": "none", "done": 0, "total": 0, "running": False}


def _copy_by_id(src_ids, src_vecs, src_ok, ids, vecs, done) -> int:
    """Copy src rows into `vecs` wherever the chunk id matches and the row isn't done; marks them done.
    `src_ok` (optional) masks src rows that were actually written."""
    import numpy as np
    if src_ids is None or not len(src_ids):
        return 0
    order = np.argsort(src_ids)
    sorted_ids = src_ids[order]
    pos = np.clip(np.searchsorted(sorted_ids, ids), 0, len(src_ids) - 1)
    hit = (sorted_ids[pos] == ids) & ~done
    if src_ok is not None:
        hit &= src_ok[order[pos]]
    dst = np.nonzero(hit)[0]
    src = order[pos[hit]]
    for i in range(0, len(dst), 200_000):                     # copy in slices to bound memory
        vecs[dst[i:i + 200_000]] = src_vecs[src[i:i + 200_000]]
    done |= hit
    return int(hit.sum())


def _load_partial(d: Path, dim: int):
    """(chunk_ids, vectors memmap) of an unfinished build that recorded its ids, or (None, None)."""
    import numpy as np
    try:
        ids = np.load(d / "chunk_ids.npy")
        vecs = np.load(d / "vectors.npy", mmap_mode="r")
        if vecs.shape == (len(ids), dim):
            return ids, vecs
    except Exception:
        pass
    return None, None


def build_index(store: Store, cfg: Config, **kw) -> dict:
    """Embed chunks -> index/{vectors.npy, chunk_ids.npy, project_ids.npy}.

    Incremental by default: vectors of chunks already in the last complete index are copied, only new
    chunks are embedded (a changed file gets new chunk ids, so it is re-embedded; deleted chunks drop out).
    Resumable: an interrupted build continues from the rows it already wrote (unless `full`), even if the
    graph changed in between: finished rows are carried over by chunk id.
    One builder at a time: raises BuildRunning while another live process holds the lock, because a
    second builder would delete the first one's folder.
    Text is fetched per batch instead of loading the whole corpus into memory. The new index is written
    to a temp folder and swapped in at the end, so an interrupted run leaves the previous index intact."""
    lock = _acquire_lock()
    try:
        return _build_index(store, cfg, **kw)
    finally:
        _release_lock(lock)


def _build_index(store: Store, cfg: Config, *, batch: int | None = None, progress=None, full: bool = False,
                 progress_every: int = 8, ann_every: int = 1_000_000) -> dict:
    import shutil

    import numpy as np
    model = _model(cfg.embed_model)
    batch = batch or _default_batch(model)
    with store.cursor() as c:
        rows = c.execute("SELECT ch.id, f.project_id FROM chunks ch JOIN files f ON f.id=ch.file_id "
                         "ORDER BY ch.id").fetchall()
    if not rows:
        return {"vectors": 0}
    out = index_dir()
    out.mkdir(parents=True, exist_ok=True)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    pids = np.array([r[1] if r[1] is not None else -1 for r in rows], dtype=np.int64)
    del rows
    dim = _dim(model)
    tmp = building_dir()
    carried = _carried_dir()
    if full and carried.exists():
        shutil.rmtree(carried)
    state = None if full or not tmp.exists() else _resume(tmp, ids, dim)
    if state is None:
        if tmp.exists():
            if not full and (tmp / "chunk_ids.npy").exists():   # the graph changed: keep its rows to carry over
                if carried.exists():
                    shutil.rmtree(carried)
                tmp.rename(carried)
            else:
                shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        vecs = np.lib.format.open_memmap(tmp / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(ids), dim))
        done = np.zeros(len(ids), dtype=bool)
    else:
        vecs, done = state
    np.save(tmp / "chunk_ids.npy", ids)
    np.save(tmp / "project_ids.npy", pids)
    resumed = int(done.sum())

    n_carried = 0
    if carried.exists():
        c_ids, c_vecs = _load_partial(carried, dim)
        if c_ids is not None:
            n_carried = _copy_by_id(c_ids, c_vecs, _filled_rows(c_vecs), ids, vecs, done)
            vecs.flush()
        del c_ids, c_vecs                                     # Windows can't delete a mapped file
        shutil.rmtree(carried, ignore_errors=True)
        if progress and n_carried:
            progress(f"carried over {n_carried} rows from the unfinished build")
    resumed += n_carried

    old_ids, old_vecs = (None, None) if full else _load_previous(out, dim)
    reused = _copy_by_id(old_ids, old_vecs, None, ids, vecs, done)
    todo = _priority(store, cfg, pids)
    todo = todo[~done[todo]]

    total, base = len(ids), int(done.sum())
    from . import ann as _ann
    ann_every = ann_every if _ann.available() else 0
    next_ann = base + ann_every if ann_every else 0
    if ann_every and base >= _ann.MIN_ROWS and not (tmp / _ann.ANN_DIR).exists():
        next_ann = base                                      # resumed build without an ANN yet: make one now
    t0 = time.time()
    _write_progress(tmp, pid=os.getpid(), done=base, total=total, embedded=0, rate=0.0, started=t0,
                    updated=t0, model=cfg.embed_model)
    for b, i in enumerate(range(0, len(todo), batch)):
        sel = np.sort(todo[i:i + batch])
        want = [int(x) for x in ids[sel]]
        ph = ",".join("?" * len(want))
        with store.cursor() as c:
            text = {r[0]: r[1] for r in c.execute(f"SELECT id, chunk_text FROM chunks WHERE id IN ({ph})", want)}
        texts = [DOC_PREFIX + (text.get(k) or "")[:EMBED_CHARS] for k in want]
        emb = model.encode(texts, normalize_embeddings=True, batch_size=batch)
        vecs[sel] = emb.astype("float16")
        done[sel] = True
        if ann_every and base + i + batch >= next_ann:
            next_ann += ann_every
            vecs.flush()
            _write_progress(tmp, pid=os.getpid(), done=base + min(i + batch, len(todo)), total=total,
                            embedded=min(i + batch, len(todo)), rate=0.0, started=t0, updated=time.time(),
                            model=cfg.embed_model, phase="refreshing ANN")
            try:
                _ann.build(tmp, rows=np.nonzero(done)[0], progress=progress)
            except Exception as e:  # noqa: BLE001 - ANN is an accelerator, never fail the build over it
                if progress:
                    progress(f"ANN refresh skipped: {e}")
        if b % progress_every == 0 or i + batch >= len(todo):
            n_emb = min(i + batch, len(todo))
            now = time.time()
            rate = n_emb / max(now - t0, 1e-6)
            if b % (progress_every * 16) == 0:
                vecs.flush()
            _write_progress(tmp, pid=os.getpid(), done=base + n_emb, total=total, embedded=n_emb,
                            rate=round(rate, 2), started=t0, updated=now, model=cfg.embed_model)
            if progress:
                progress(f"embedded {n_emb}/{len(todo)} new chunks ({reused} reused, {resumed} resumed)")
    vecs.flush()
    del vecs, state, old_vecs                                 # Windows can't rename dirs with open maps
    if _ann.available() and total >= _ann.MIN_ROWS:
        try:
            _ann.build(tmp, rows=np.arange(total), progress=progress)
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"ANN build skipped: {e}")
    (tmp / "progress.json").unlink(missing_ok=True)
    _swap_in(tmp, out)
    return {"vectors": len(ids), "embedded": int(len(todo)), "reused": reused, "resumed": resumed,
            "carried": n_carried, "dim": dim, "dir": str(out)}


def _swap_in(tmp: Path, out: Path, attempts: int = 10) -> None:
    """Replace `out` with the finished `tmp`. On Windows a process that still maps the old index (an MCP
    server, say) blocks the rename; retry briefly, then restore the old index and say what to do. The
    finished build stays in place, so re-running `carto index` only has to swap."""
    import gc
    import shutil
    old = out.parent / (out.name + ".old")
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    for i in range(attempts):
        gc.collect()
        try:
            if out.exists():
                out.rename(old)
            try:
                tmp.rename(out)
            except OSError:
                if old.exists() and not out.exists():
                    old.rename(out)
                raise
            shutil.rmtree(old, ignore_errors=True)
            return
        except OSError:
            if i == attempts - 1:
                raise RuntimeError(f"the new index is complete in {tmp} but {out} is in use by another process "
                                   "(close MCP servers / `carto viz`, then run `carto index` again to swap it in)")
            time.sleep(0.5)


class SemanticIndex:
    """The complete index, or — while the first build is still running — the partial one. Rows not yet
    embedded are zero vectors, so they score 0 and never outrank real matches."""

    def __init__(self, cfg: Config) -> None:
        import numpy as np
        d = index_dir()
        self.building = not _complete(d)
        if self.building:
            d = building_dir()
        self.dir = d
        self.vectors = np.load(d / "vectors.npy", mmap_mode="r")
        self.chunk_ids = np.load(d / "chunk_ids.npy")
        self.project_ids = np.load(d / "project_ids.npy")
        self.cfg = cfg
        self.ann = None
        from .ann import Ann
        self._ann_ok = Ann.usable(d, len(self.chunk_ids))
        self.signature = SemanticIndex.current_signature()

    @property
    def mode(self) -> str:
        return "ann" if self._ann_ok else "exact"

    def _load_ann(self) -> None:
        if self._ann_ok and self.ann is None:
            from .ann import Ann
            try:
                self.ann = Ann(self.dir)
            except Exception:
                self._ann_ok = False

    @staticmethod
    def exists() -> bool:
        return _complete(index_dir()) or _complete(building_dir())

    @staticmethod
    def current_signature() -> tuple:
        """Changes when the index is swapped in or its ANN layer is rebuilt (long-lived servers reload)."""
        out = []
        for d in (index_dir(), building_dir()):
            for name in ("chunk_ids.npy", "ann.json"):
                try:
                    out.append((d / name).stat().st_mtime)
                except OSError:
                    out.append(None)
        return tuple(out)

    def _exact_top(self, qv, fetch: int, block: int = 262_144):
        import numpy as np
        n = self.vectors.shape[0]
        idx_parts, sim_parts = [], []
        for i in range(0, n, block):
            s = np.asarray(self.vectors[i:i + block], dtype=np.float32) @ qv
            k = min(fetch, s.shape[0])
            top = np.argpartition(-s, k - 1)[:k]
            idx_parts.append(top + i)
            sim_parts.append(s[top])
        idx, sims = np.concatenate(idx_parts), np.concatenate(sim_parts)
        order = np.argsort(-sims)[:fetch]
        return idx[order], sims[order]

    def _qv(self, query: str):
        import numpy as np
        last = getattr(self, "_last_q", None)
        if last is None or last[0] != query:
            qv = _model(self.cfg.embed_model).encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
            self._last_q = last = (query, np.asarray(qv, dtype=np.float32))
        return last[1]

    def search_in_projects(self, query: str, project_ids: list[int], top_k: int = 30) -> list[int]:
        """Exact search restricted to the given projects (for queries that name a project)."""
        import numpy as np
        try:
            rows = np.flatnonzero(np.isin(self.project_ids, project_ids))
            if not len(rows):
                return []
            sims = np.asarray(self.vectors[rows], dtype=np.float32) @ self._qv(query)
            if self.building:
                rows, sims = rows[sims > 0], sims[sims > 0]
            order = np.argsort(-sims)[:top_k]
        except Exception:
            return []
        return [int(self.chunk_ids[rows[i]]) for i in order]

    def _ranked(self, query: str, fetch: int):
        import numpy as np
        qv = self._qv(query)
        fetch = min(fetch, self.vectors.shape[0])
        self._load_ann()
        if self.ann is not None:
            cand = np.sort(self.ann.candidates(qv, max(fetch, 2000)))
            sims = np.asarray(self.vectors[cand], dtype=np.float32) @ qv
            order = np.argsort(-sims)[:fetch]
            idx, sims = cand[order], sims[order]
        else:
            idx, sims = self._exact_top(qv, fetch)
        if self.building:
            idx = idx[sims > 0]
        return idx

    def search_chunk_ids(self, query: str, top_k: int = 30, fetch: int = 400) -> list[int]:
        try:
            idx = self._ranked(query, fetch)
        except Exception:
            return []
        return [int(self.chunk_ids[i]) for i in idx[:top_k]]
