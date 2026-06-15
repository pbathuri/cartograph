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
        except Exception:
            pass
        _MODEL = SentenceTransformer(name, trust_remote_code=True, device=dev)
        try:
            _MODEL.max_seq_length = 192  # speed; chunks are short
        except Exception:
            pass
    return _MODEL


def build_index(store: Store, cfg: Config, *, batch: int = 256, progress=None) -> dict:
    """Embed every chunk -> index/{vectors.npy, chunk_ids.npy, project_ids.npy}."""
    import numpy as np
    model = _model(cfg.embed_model)
    rows = []
    with store.cursor() as c:
        rows = c.execute(
            "SELECT ch.id, ch.chunk_text, f.project_id FROM chunks ch JOIN files f ON f.id=ch.file_id"
        ).fetchall()
    if not rows:
        return {"vectors": 0}
    out = index_dir()
    out.mkdir(parents=True, exist_ok=True)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    pids = np.array([r[2] if r[2] is not None else -1 for r in rows], dtype=np.int64)
    dim = model.get_sentence_embedding_dimension()
    vecs = np.lib.format.open_memmap(out / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(rows), dim))
    for i in range(0, len(rows), batch):
        texts = [DOC_PREFIX + (r[1] or "")[:600] for r in rows[i:i + batch]]
        emb = model.encode(texts, normalize_embeddings=True, batch_size=batch)
        vecs[i:i + len(texts)] = emb.astype("float16")
        if progress and i % (batch * 8) == 0:
            progress(f"embedded {i + len(texts)}/{len(rows)} chunks")
    vecs.flush()
    np.save(out / "chunk_ids.npy", ids)
    np.save(out / "project_ids.npy", pids)
    return {"vectors": len(rows), "dim": dim, "dir": str(out)}


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
