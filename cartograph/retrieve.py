"""Hybrid retrieval — the headline capability. Reciprocal Rank Fusion of full-semantic + FTS.

In the engine Cartograph distills from, hybrid measured success@10 0.986 vs 0.958 for either tier
alone (n=71): it's never worse, because it surfaces hits from BOTH tiers — semantic catches
paraphrases ("validate inputs" -> pydantic), FTS catches exact/rare tokens ("dvc", a function name).
Works in pure-FTS mode with zero ML installed; turns on semantic automatically once you build an index.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config
from .storage import Store

_FTS_BAD = re.compile(r"[^\w\s]")


def sanitize(q: str) -> str:
    toks = [t for t in _FTS_BAD.sub(" ", q).split() if len(t) >= 2]
    return " OR ".join(f'"{t}"' for t in toks)


def rrf_fuse(rankings: list[list[Any]], top_k: int, k: int = 60) -> list[Any]:
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            if key is not None:
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [key for key, _ in sorted(scores.items(), key=lambda t: t[1], reverse=True)][:top_k]


@dataclass
class RetrieveResult:
    query: str
    method: str
    projects: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_INDEX = None


def _semantic_chunk_ids(cfg: Config, query: str, top_k: int) -> list[int]:
    global _INDEX
    from .embed import SemanticIndex
    if not (available_semantic() and SemanticIndex.exists()):
        return []
    try:
        if _INDEX is None:
            _INDEX = SemanticIndex(cfg)
        return _INDEX.search_chunk_ids(query, top_k=top_k)
    except Exception:
        return []


def available_semantic() -> bool:
    from .embed import available
    return available()


def _chunks_by_id(store: Store, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    sql = ("SELECT ch.id AS chunk_id, ch.chunk_text, f.path AS file_path, p.name AS project_name "
           "FROM chunks ch JOIN files f ON f.id=ch.file_id LEFT JOIN projects p ON p.id=f.project_id "
           f"WHERE ch.id IN ({ph})")
    with store.cursor() as c:
        return {int(dict(r)["chunk_id"]): dict(r) for r in c.execute(sql, ids).fetchall()}


def retrieve(query: str, store: Store, cfg: Config, *, top_k: int = 10) -> RetrieveResult:
    """Hybrid chunk retrieval (RRF of semantic + FTS), collapsed to ranked projects + the snippets."""
    san = sanitize(query)
    fts_rows = store.search_chunks(san, limit=top_k * 3) if san else []
    sem_ids = _semantic_chunk_ids(cfg, query, top_k * 3)
    sem_rows = _chunks_by_id(store, sem_ids)
    pool = {r["chunk_id"]: r for r in fts_rows}
    pool.update(sem_rows)
    fused = rrf_fuse([sem_ids, [r["chunk_id"] for r in fts_rows]], top_k)
    chunks = [pool[i] for i in fused if i in pool]
    method = "hybrid" if sem_ids else "fts"
    projects: list[str] = []
    for ch in chunks:
        nm = ch.get("project_name")
        if nm and nm not in projects:
            projects.append(nm)
    return RetrieveResult(query=query, method=method, projects=projects, chunks=chunks)
