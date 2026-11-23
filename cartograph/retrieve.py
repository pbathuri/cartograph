"""Hybrid retrieval — the headline capability. Reciprocal Rank Fusion of full-semantic + FTS.

In the engine Cartograph distills from, hybrid measured success@10 0.986 vs 0.958 for either tier
alone (n=71): it's never worse, because it surfaces hits from BOTH tiers — semantic catches
paraphrases ("validate inputs" -> pydantic), FTS catches exact/rare tokens ("dvc", a function name).
Works in pure-FTS mode with zero ML installed; turns on semantic automatically once you build an index.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config
from .documents import cite, legacy_locator
from .storage import Store

_FTS_BAD = re.compile(r"[^\w\s]")


_COMPOUND = re.compile(r"\b\w+(?:[-./:]\w+)+\b")


def sanitize(q: str) -> str:
    """FTS5 query: every word as an OR term, plus identifiers like IL-6, KX-417, CVE-2024-1234 or
    torch.nn as exact phrases, so their pieces matching elsewhere don't outrank the real thing."""
    toks = [t for t in _FTS_BAD.sub(" ", q).split() if len(t) >= 2]
    phrases = [" ".join(_FTS_BAD.sub(" ", m).split()) for m in _COMPOUND.findall(q)]
    terms = [f'"{t}"' for t in toks] + [f'"{p}"' for p in dict.fromkeys(phrases) if " " in p]
    return " OR ".join(terms)


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
    if not (SemanticIndex.exists() and available_semantic()):
        return []
    try:
        if _INDEX is None or _INDEX.signature != SemanticIndex.current_signature():
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
    sql = (f"SELECT ch.id AS chunk_id, ch.chunk_text, {store.locator_sql()} AS locator, f.path AS file_path, "
           "p.name AS project_name, p.id AS project_id "
           "FROM chunks ch JOIN files f ON f.id=ch.file_id LEFT JOIN projects p ON p.id=f.project_id "
           f"WHERE ch.id IN ({ph})")
    with store.cursor() as c:
        return {int(dict(r)["chunk_id"]): dict(r) for r in c.execute(sql, ids).fetchall()}


def retrieve(query: str, store: Store, cfg: Config, *, top_k: int = 10, rrf_k: int | None = None) -> RetrieveResult:
    """Hybrid chunk retrieval (RRF of semantic + FTS), collapsed to ranked projects + the snippets.
    `rrf_k` defaults to the tuned value (`carto improve`), else 60."""
    if rrf_k is None:
        from .tuning import load as _tuned
        rrf_k = int(_tuned().get("rrf_k", 60))
    san = sanitize(query)
    fts_rows = store.search_chunks(san, limit=top_k * 3) if san else []
    sem_ids = _semantic_chunk_ids(cfg, query, top_k * 3)
    sem_rows = _chunks_by_id(store, sem_ids)
    pool = {r["chunk_id"]: r for r in fts_rows}
    pool.update(sem_rows)
    # fused RRF scores (kept, not just the order) so downstream can measure relevance MARGIN/ambiguity.
    fscore: dict[Any, float] = {}
    for ranking in (sem_ids, [r["chunk_id"] for r in fts_rows]):
        for rank, key in enumerate(ranking, 1):
            if key is not None:
                fscore[key] = fscore.get(key, 0.0) + 1.0 / (rrf_k + rank)
    fused = [k for k, _ in sorted(fscore.items(), key=lambda t: t[1], reverse=True)][:top_k]
    chunks = []
    for i in fused:
        if i in pool:
            ch = dict(pool[i])
            ch["score"] = round(fscore.get(i, 0.0), 6)       # fused relevance score
            if not ch.get("locator"):
                ch["locator"] = legacy_locator(ch.get("file_path") or "", ch.get("chunk_text") or "")
            ch["cite"] = cite(ch.get("file_path") or "", ch.get("locator"))
            chunks.append(ch)
    method = "hybrid" if sem_ids else "fts"
    projects: list[str] = []
    for ch in chunks:
        nm = ch.get("project_name")
        if nm and nm not in projects:
            projects.append(nm)
    return RetrieveResult(query=query, method=method, projects=projects, chunks=chunks)


# ---------------------------------------------------------------------------------------------- context
MODES = ("mix", "chunks", "projects", "graph", "memory")
_STOP = {"what", "which", "where", "when", "with", "from", "that", "this", "have", "does", "about", "into",
         "using", "used", "uses", "use", "project", "projects", "file", "files", "code", "show", "find",
         "how", "the", "and", "for", "are", "any", "all", "my", "our", "your", "cite", "cites", "paper"}


def tokens(text: str) -> int:
    """Rough token count (~4 characters per token) used for budgets."""
    return (len(text) + 3) // 4


def _fit(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("\n\n", "\n", ". "):
        j = cut.rfind(sep)
        if j > max_chars * 0.6:
            return cut[:j + (1 if sep == ". " else 0)].rstrip() + " …"
    return cut.rstrip() + " …"


def _dedupe(chunks: list[dict], threshold: float = 0.6) -> list[dict]:
    """Drop chunks that mostly repeat a better-ranked chunk of the same file (overlapping windows)."""
    kept: list[tuple[str, set]] = []
    out = []
    for ch in chunks:
        toks = set(re.findall(r"\w{3,}", (ch.get("chunk_text") or "").lower()))
        fp = ch.get("file_path")
        if any(f == fp and len(toks & t) / max(1, min(len(toks), len(t))) >= threshold for f, t in kept):
            continue
        kept.append((fp, toks))
        out.append(ch)
    return out


def _graph_facts(store: Store, query: str, limit: int) -> list[dict]:
    from . import entities as E
    terms = [t for t in re.findall(r"[\w.\-/]{3,}", query.lower()) if t not in _STOP]
    seen, out = set(), []
    for t in sorted(terms, key=len, reverse=True)[:6]:
        for f in E.find(store, t, limit=limit):
            key = (f["entity"], f.get("project"), f["relation"])
            if key not in seen:
                seen.add(key)
                out.append(f)
    out.sort(key=lambda f: (-f["files"], -f["mentions"]))
    return out[:limit]


def context(query: str, store: Store, cfg: Config, *, mode: str = "mix", max_tokens: int = 2000,
            top_k: int = 8, project: str | None = None, cwd: str | None = None) -> dict:
    """Everything an agent needs for `query`, under a hard token budget, every item citable.

    mix      memories (decisions, rejected approaches...) + entity facts + code/doc chunks
    chunks   hybrid-retrieved snippets only          projects  ranked projects with their best hit
    graph    entity facts (deps, papers, repos...)   memory    agent memories only"""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    budget = max(200, int(max_tokens)) * 4
    used = 0
    out: dict[str, Any] = {"query": query, "mode": mode, "max_tokens": int(max_tokens)}

    if mode in ("mix", "memory") and store.has_table("memories"):
        from . import memory as M
        share = budget if mode == "memory" else budget // 5
        mem = M.recall(store, query, project=project, cwd=cwd, limit=12, max_chars=share)
        if not mem["memories"] and (project or cwd):
            mem = M.recall(store, query, limit=12, max_chars=share)
        items = [{k: m[k] for k in ("id", "kind", "text", "project", "created", "stale") if m.get(k)}
                 for m in mem["memories"]]
        out["memories"] = items
        used += sum(len(m["text"]) + 40 for m in items)

    if mode in ("mix", "graph"):
        share = budget - used if mode == "graph" else budget // 6
        facts, size = [], 0
        for f in _graph_facts(store, query, limit=30 if mode == "graph" else 8):
            line = len(json.dumps(f))
            if size + line > share and facts:
                break
            facts.append(f)
            size += line
        out["facts"] = facts
        used += size

    if mode in ("mix", "chunks", "projects"):
        res = retrieve(query, store, cfg, top_k=top_k * 2)
        out["method"] = res.method
        chunks = _dedupe(res.chunks)
        if mode == "projects":
            best: dict[str, dict] = {}
            for ch in chunks:
                nm = ch.get("project_name") or "?"
                b = best.setdefault(nm, {"project": nm, "hits": 0, "best": ch.get("cite"), "score": ch["score"]})
                b["hits"] += 1
            out["projects"] = list(best.values())[:top_k]
            used += len(json.dumps(out["projects"]))
        else:
            room = budget - used
            picked = []
            for ch in chunks[:top_k]:
                if room < 160:
                    break
                text = _fit(ch.get("chunk_text") or "", min(room - 80, max(400, room // 2)))
                picked.append({"cite": ch.get("cite"), "project": ch.get("project_name"),
                               "score": ch.get("score"), "text": text})
                room -= len(text) + 80
            out["chunks"] = picked
            used = budget - room
    def size() -> int:
        return tokens(json.dumps({k: v for k, v in out.items() if k in ("memories", "facts", "chunks", "projects")}))
    while size() > max_tokens:                               # hard cap: drop lowest-ranked items last-in first
        for key in ("chunks", "projects", "facts", "memories"):
            if out.get(key):
                out[key].pop()
                break
        else:
            break
    out["tokens"] = size()
    return out
