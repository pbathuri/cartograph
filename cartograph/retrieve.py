"""Hybrid retrieval — the headline capability. Reciprocal Rank Fusion of full-semantic + FTS.

In the engine Cartograph distills from, hybrid measured success@10 0.986 vs 0.958 for either tier
alone (n=71): it's never worse, because it surfaces hits from BOTH tiers — semantic catches
paraphrases ("validate inputs" -> pydantic), FTS catches exact/rare tokens ("dvc", a function name).
Works in pure-FTS mode with zero ML installed; turns on semantic automatically once you build an index.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config
from .documents import cite, legacy_locator
from .storage import Store

_FTS_BAD = re.compile(r"[^\w\s]")


_COMPOUND = re.compile(r"\b\w+(?:[-./:]\w+)+\b")

# Question words and glue. Each one matches millions of chunks: on a 7M-chunk graph they made keyword
# search 2-5x slower while adding nothing to the ranking.
STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing done down during each else few for from further get gets got had has have
having he her here hers him his how i if in into is it its itself just me more most my myself no nor not now
of off on once only or other our ours out over own same she should so some such than that the their theirs
them then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself
""".split())


def keywords(q: str) -> list[str]:
    """The query's searchable words: punctuation stripped, stopwords dropped (all kept if that empties it)."""
    toks = [t for t in _FTS_BAD.sub(" ", q).split() if len(t) >= 2]
    kept = [t for t in toks if t.lower() not in STOPWORDS]
    return kept or toks


def sanitize(q: str) -> str:
    """FTS5 query: every keyword as an OR term, plus identifiers like IL-6, KX-417, CVE-2024-1234 or
    torch.nn as exact phrases, so their pieces matching elsewhere don't outrank the real thing."""
    toks = keywords(q)
    phrases = [" ".join(_FTS_BAD.sub(" ", m).split()) for m in _COMPOUND.findall(q)]
    terms = [f'"{t}"' for t in dict.fromkeys(toks)] + [f'"{p}"' for p in dict.fromkeys(phrases) if " " in p]
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
_INDEX_LOCK = threading.Lock()


def _semantic_chunk_ids(cfg: Config, query: str, top_k: int) -> list[int]:
    global _INDEX
    from .embed import SemanticIndex
    if not (SemanticIndex.exists() and available_semantic()):
        return []
    try:
        with _INDEX_LOCK:                    # the MCP server warms this up on a thread at start
            if _INDEX is None or _INDEX.signature != SemanticIndex.current_signature():
                _INDEX = SemanticIndex(cfg)
            return _INDEX.search_chunk_ids(query, top_k=top_k)
    except Exception:
        return []


def _semantic_in_projects(cfg: Config, query: str, project_ids: list[int], top_k: int) -> list[int]:
    if not _semantic_chunk_ids(cfg, query, 1):                # loads (or reloads) the index
        return []
    with _INDEX_LOCK:
        return _INDEX.search_in_projects(query, project_ids, top_k=top_k)


def _short_name(project: str) -> str | None:
    """The word that names a project, if its folder name looks like a name and not a plain word:
    a GitHub-style 'owner__repo' clone ('pallets__flask' -> 'flask') or a name with inner capitals or
    digits ('nanoGPT', 'Kronos2'). Plain folders like 'Notes', 'outputs' or 'data' name nothing."""
    leaf = project.rsplit("/", 1)[-1]
    owner, sep, repo = leaf.partition("__")
    short = repo if sep and owner else leaf
    if not (sep and owner) and not re.search(r".[A-Z]|\d", short):
        return None
    short = short.lower()
    if len(short) < 4 or short in STOPWORDS or short in _STOP:
        return None
    return short


def named_projects(store: Store, query: str) -> dict[int, str]:
    """Projects the query names ("... in nanoGPT", "how does requests ..."): id -> the word that named it."""
    words = {w.lower() for w in keywords(query)}
    with store.cursor() as c:
        rows = c.execute("SELECT id, name FROM projects").fetchall()
    out = {}
    for pid, name in rows:
        short = _short_name(name or "")
        if short and short in words:
            out[pid] = short
    return dict(list(out.items())[:3])


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


NAMED_WEIGHT = 2.0          # RRF weight of the list searched inside a project the query names


def retrieve(query: str, store: Store, cfg: Config, *, top_k: int = 10, rrf_k: int | None = None) -> RetrieveResult:
    """Hybrid chunk retrieval (RRF of semantic + FTS), collapsed to ranked projects + the snippets.
    `rrf_k` defaults to the tuned value (`carto improve`), else 60."""
    if rrf_k is None:
        from .tuning import load as _tuned
        rrf_k = int(_tuned().get("rrf_k", 60))
    san = sanitize(query)
    fts_rows = store.search_chunks(san, limit=top_k * 3) if san else []
    sem_ids = _semantic_chunk_ids(cfg, query, top_k * 3)
    rankings = [sem_ids, [r["chunk_id"] for r in fts_rows]]
    weights = [1.0, 1.0]
    named = named_projects(store, query)
    if named:
        # The project's own files rarely contain its name, so search inside it as well. The vector
        # search is fast; keyword search inside one project scores every match in the graph, so it only
        # runs when there is no vector index.
        inside_ids = _semantic_in_projects(cfg, query, list(named), top_k * 3) if sem_ids else []
        if not sem_ids:
            rest = sanitize(" ".join(w for w in keywords(query) if w.lower() not in set(named.values())) or query)
            inside = [r for pid in named for r in store.search_chunks(rest, limit=top_k * 3, project_id=pid)] if rest else []
            fts_rows += inside
            inside_ids = [r["chunk_id"] for r in inside]
        rankings.append(inside_ids)
        weights.append(NAMED_WEIGHT)
    sem_rows = _chunks_by_id(store, sem_ids + (rankings[2] if len(rankings) > 2 and sem_ids else []))
    pool = {r["chunk_id"]: r for r in fts_rows}
    pool.update(sem_rows)
    # fused RRF scores (kept, not just the order) so downstream can measure relevance MARGIN/ambiguity.
    fscore: dict[Any, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, key in enumerate(ranking, 1):
            if key is not None:
                fscore[key] = fscore.get(key, 0.0) + w / (rrf_k + rank)
    fused = [k for k, _ in sorted(fscore.items(), key=lambda t: t[1], reverse=True)]
    chunks, seen = [], set()
    for i in fused:
        if len(chunks) >= top_k:
            break
        if i in pool:
            body = " ".join((pool[i].get("chunk_text") or "").split())
            if body in seen:                 # the same passage in a second checkout or copy of a file
                continue
            seen.add(body)
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


def _on_topic(query: str, text: str, kind: str) -> bool:
    """Whether a memory belongs in a mixed answer. Curated memories (decisions, rejections...) need one
    shared keyword; auto-captured session logs need half the query's keywords, because a long session
    log contains nearly any single word."""
    words = {w.lower() for w in keywords(query) if w.lower() not in _STOP}
    if not words:
        return True
    have = {w[:6] for w in re.findall(r"\w{2,}", text.lower())}
    shared = sum(w[:6] in have for w in words)
    return shared >= len(words) / 2 if kind == "session" else shared >= 1


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
                 for m in mem["memories"] if mode == "memory" or _on_topic(query, m["text"], m["kind"])]
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
