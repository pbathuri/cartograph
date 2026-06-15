"""Field router — a LEARNED, corpus-derived replacement for keyword field detection.

The catalog's keyword `match_field` only knows fields it has keywords for. The router instead fits one
centroid embedding PER field from the user's own ingested chunks, then routes a prompt to its field by
nearest centroid. That makes field detection agnostic to *any* field the user actually works in (not
just the built-in list), and grounded in their real vocabulary. Trained by `carto train`; persisted in
the workspace. Degrades to keyword matching when embeddings aren't installed or no centroids exist.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config, home
from .elite.catalog import match_field


def _centroids_path() -> Path:
    return home() / "field_centroids.npz"


def build_centroids(store, cfg: Config, *, per_field: int = 400, progress=None) -> dict:
    """Fit one mean-embedding centroid per field from sampled chunks. Returns {field: count}."""
    try:
        import numpy as np

        from .embed import DOC_PREFIX, _model, available
        if not available():
            return {}
        # gather up to per_field chunk texts per field
        by_field: dict[str, list[str]] = {}
        with store.cursor() as c:
            rows = c.execute(
                "SELECT p.field AS field, ch.chunk_text AS t FROM chunks ch "
                "JOIN files f ON f.id=ch.file_id JOIN projects p ON p.id=f.project_id "
                "WHERE p.field IS NOT NULL AND p.field != 'general'").fetchall()
        for r in rows:
            d = dict(r)
            lst = by_field.setdefault(d["field"], [])
            if len(lst) < per_field and d["t"]:
                lst.append(DOC_PREFIX + d["t"][:600])
        if not by_field:
            return {}
        model = _model(cfg.embed_model)
        cents, counts = {}, {}
        for fld, texts in by_field.items():
            if progress:
                progress(f"centroid: {fld} ({len(texts)} chunks)")
            emb = model.encode(texts, normalize_embeddings=True, batch_size=128)
            v = np.asarray(emb).mean(axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                cents[fld] = (v / n).astype("float32")
                counts[fld] = len(texts)
        if cents:
            np.savez(_centroids_path(), **cents)
        return counts
    except Exception:
        return {}


_CENTS = None


def _load_centroids():
    global _CENTS
    if _CENTS is not None:
        return _CENTS
    if not _centroids_path().exists():
        _CENTS = {}
        return _CENTS
    try:
        import numpy as np
        with np.load(_centroids_path()) as z:
            _CENTS = {k: z[k] for k in z.files}
    except Exception:
        _CENTS = {}
    return _CENTS


def route(prompt: str, cfg: Config, *, min_score: float = 0.18) -> dict:
    """Route a prompt to a field. Learned (nearest centroid) when available + confident, else keyword."""
    cents = _load_centroids()
    if cents:
        try:
            import numpy as np

            from .embed import QUERY_PREFIX, _model, available
            if available():
                qv = np.asarray(_model(cfg.embed_model).encode([QUERY_PREFIX + prompt],
                                                               normalize_embeddings=True)[0])
                scored = sorted(((float(np.dot(qv, v)), f) for f, v in cents.items()), reverse=True)
                top_s, top_f = scored[0]
                if top_s >= min_score:
                    return {"field": top_f, "score": round(top_s, 3), "method": "learned",
                            "runner_up": (scored[1][1] if len(scored) > 1 else None)}
        except Exception:
            pass
    kw = match_field(prompt)
    return {"field": kw or "general", "score": None, "method": "keyword"}


def infer_field(prompt: str, cfg: Config) -> str:
    """Convenience: the routed field (learned if possible, else keyword)."""
    return route(prompt, cfg)["field"]
