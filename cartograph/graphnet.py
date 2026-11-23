"""Graph network layer — project graph + personalized PageRank (PPR).

Retrieval scores chunks one at a time. Real work is relational: the Blink analysis leans on the EOG tool,
the CUDA homework on the HPC course notes, a time-series model on the backtesting repo. This module turns
the project table into a weighted graph and lets relevance FLOW along it:

  edges (undirected, weighted, stored in `edges` with relation + weight metadata)
    PROJECT_SIMILAR   cosine of project centroids from the semantic index (kNN, k=8, cos >= 0.55)
    PROJECT_COACTIVE  projects touched in the same AutoLog activity summary (count, log-scaled)
    PROJECT_RELATED_TO_PROJECT  shared inferred field (existing, weak)

  activity prior  recency-weighted AutoLog touches per project (half-life 14 days)

  PPR             teleport = mix(retrieval seed scores, activity prior); power iteration on the
                  row-normalized weighted adjacency. A project scores high when it is relevant itself
                  OR sits next to relevant, recently-active work. Used as a re-ranking feature (weight
                  tuned by `carto improve`) and to surface "related threads" the query didn't name.

Pure NumPy/stdlib. Everything degrades: no index -> no SIMILAR edges; no AutoLog -> no COACTIVE edges.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from .config import Config, home
from .storage import Store

AUTOLOG_DB = Path.home() / "Library/Application Support/ContextD/contextd.sqlite"
SIM_K = 8
SIM_MIN = 0.55
HALF_LIFE_DAYS = 14.0
_REL_W = {"PROJECT_SIMILAR": 1.0, "PROJECT_COACTIVE": 1.0, "PROJECT_RELATED_TO_PROJECT": 0.15}


def _graph_path() -> Path:
    return home() / "project_graph.json"


# ----------------------------------------------------------------------------------------- mapping
def file_index(store: Store) -> tuple[dict[str, int], dict[str, int]]:
    """(last-two-components -> project_id, basename -> project_id), basename only when unambiguous.
    Used to map AutoLog documentPaths (often partial) onto graph projects."""
    tail2: dict[str, int] = {}
    base: dict[str, set[int]] = defaultdict(set)
    with store.cursor() as c:
        for r in c.execute("SELECT path, project_id FROM files WHERE project_id IS NOT NULL"):
            p = (r["path"] or "").replace("\\", "/")
            parts = p.rsplit("/", 2)
            if len(parts) >= 2:
                tail2["/".join(parts[-2:]).lower()] = r["project_id"]
            base[parts[-1].lower()].add(r["project_id"])
    uniq = {k: next(iter(v)) for k, v in base.items() if len(v) == 1 and len(k) > 4}
    return tail2, uniq


def map_doc_paths(paths: list[str], tail2: dict[str, int], base: dict[str, int]) -> set[int]:
    out: set[int] = set()
    for raw in paths or []:
        p = str(raw).replace("\\", "/").strip().rstrip("/")
        if not p:
            continue
        parts = p.rsplit("/", 2)
        key2 = "/".join(parts[-2:]).lower() if len(parts) >= 2 else ""
        if key2 in tail2:
            out.add(tail2[key2])
        elif parts[-1].lower() in base:
            out.add(base[parts[-1].lower()])
    return out


def read_activity(db: Path | None = None, *, limit: int = 200_000) -> list[dict]:
    """AutoLog summaries with the files they touched. Professional lanes only (no messaging/watching)."""
    path = db or AUTOLOG_DB
    if not Path(path).exists():
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, startTimestamp AS ts, summary, keyTopics, documentPaths, activityType FROM summaries "
            "WHERE documentPaths IS NOT NULL AND documentPaths NOT IN ('', '[]') "
            "AND COALESCE(activityType,'') NOT IN ('communication','watching','other') "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        con.close()
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        try:
            docs = json.loads(r["documentPaths"] or "[]")
            topics = json.loads(r["keyTopics"] or "[]")
        except (TypeError, ValueError):
            continue
        ts = float(r["ts"] or 0)
        if ts > 1e11:                                       # ms -> s
            ts /= 1000.0
        out.append({"id": r["id"], "ts": ts, "summary": r["summary"] or "", "topics": topics,
                    "docs": docs, "type": r["activityType"] or ""})
    return out


# ------------------------------------------------------------------------------------------ build
def _project_centroids(store: Store):
    """Mean normalized vector per project from the semantic index (streamed). {} without an index."""
    try:
        import numpy as np

        from .config import index_dir
        d = index_dir()
        vecs = np.load(d / "vectors.npy", mmap_mode="r")
        pids = np.load(d / "project_ids.npy")
    except Exception:
        return {}
    if not len(pids):
        return {}
    uniq, inv = np.unique(pids, return_inverse=True)
    sums = np.zeros((len(uniq), vecs.shape[1]), dtype=np.float32)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float32)
    step = 200_000
    for i in range(0, len(pids), step):
        np.add.at(sums, inv[i:i + step], np.asarray(vecs[i:i + step], dtype=np.float32))
    sums /= np.maximum(counts, 1)[:, None]
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    sums /= np.maximum(norms, 1e-9)
    return {int(p): sums[j] for j, p in enumerate(uniq) if p >= 0}


def build_graph(store: Store, cfg: Config, *, autolog_db: Path | None = None, now: float | None = None,
                progress=None) -> dict:
    """Recompute SIMILAR + COACTIVE edges and the activity prior; persist to the graph + project_graph.json."""
    now = now or time.time()
    edges: dict[tuple[int, int, str], float] = {}

    cents = _project_centroids(store)
    if len(cents) >= 2:
        import numpy as np
        ids = list(cents)
        M = np.stack([cents[i] for i in ids])
        S = M @ M.T
        np.fill_diagonal(S, -1)
        for a, row in enumerate(S):
            for b in np.argsort(-row)[:SIM_K]:
                if row[b] >= SIM_MIN:
                    key = (min(ids[a], ids[b]), max(ids[a], ids[b]), "PROJECT_SIMILAR")
                    edges[key] = max(edges.get(key, 0.0), float(row[b]))
        if progress:
            progress(f"similarity edges from {len(ids)} project centroids")

    tail2, base = file_index(store)
    prior: dict[int, float] = defaultdict(float)
    co: dict[tuple[int, int], int] = defaultdict(int)
    acts = read_activity(autolog_db)
    for a in acts:
        ps = sorted(map_doc_paths(a["docs"], tail2, base))
        age_days = max(0.0, (now - a["ts"]) / 86400.0) if a["ts"] else 365.0
        w = 0.5 ** (age_days / HALF_LIFE_DAYS)
        for p in ps:
            prior[p] += w
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                co[(ps[i], ps[j])] += 1
    for (a, b), n in co.items():
        edges[(a, b, "PROJECT_COACTIVE")] = round(min(1.0, math.log1p(n) / math.log(20)), 4)
    if progress:
        progress(f"co-activity edges from {len(acts)} AutoLog summaries")

    with store.cursor() as c:
        c.execute("DELETE FROM edges WHERE relation IN ('PROJECT_SIMILAR','PROJECT_COACTIVE')")
        c.executemany("INSERT OR REPLACE INTO edges(src_id, relation, dst_id, metadata) VALUES(?,?,?,?)",
                      [(a, rel, b, json.dumps({"w": round(w, 4)})) for (a, b, rel), w in edges.items()])
    tot = sum(prior.values()) or 1.0
    data = {"built": now, "prior": {str(k): round(v / tot, 6) for k, v in prior.items()},
            "n_similar": sum(1 for k in edges if k[2] == "PROJECT_SIMILAR"),
            "n_coactive": sum(1 for k in edges if k[2] == "PROJECT_COACTIVE"),
            "n_activity": len(acts)}
    _graph_path().parent.mkdir(parents=True, exist_ok=True)
    tmp = _graph_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, _graph_path())
    reset_cache()
    return {k: v for k, v in data.items() if k != "prior"} | {"projects_with_activity": len(prior)}


# ------------------------------------------------------------------------------------------ query
class ProjectGraph:
    def __init__(self, adj: dict[int, dict[int, float]], prior: dict[int, float]):
        self.adj = adj
        self.prior = prior

    @classmethod
    def load(cls, store: Store) -> "ProjectGraph":
        adj: dict[int, dict[int, float]] = defaultdict(dict)
        with store.cursor() as c:
            for r in c.execute("SELECT src_id, relation, dst_id, metadata FROM edges "
                               "WHERE relation IN ('PROJECT_SIMILAR','PROJECT_COACTIVE','PROJECT_RELATED_TO_PROJECT')"):
                try:
                    w = float(json.loads(r["metadata"] or "{}").get("w", 1.0))
                except (TypeError, ValueError, AttributeError):
                    w = 1.0
                w *= _REL_W.get(r["relation"], 0.1)
                a, b = int(r["src_id"]), int(r["dst_id"])
                adj[a][b] = adj[a].get(b, 0.0) + w
                adj[b][a] = adj[b].get(a, 0.0) + w
        prior: dict[int, float] = {}
        try:
            prior = {int(k): float(v) for k, v in json.loads(_graph_path().read_text())["prior"].items()}
        except Exception:
            pass
        return cls(dict(adj), prior)

    def ppr(self, seeds: dict[int, float], *, damping: float = 0.85, prior_mix: float = 0.3,
            iters: int = 100, tol: float = 1e-7) -> dict[int, float]:
        """Personalized PageRank. Teleport = (1-prior_mix)*seeds + prior_mix*activity prior."""
        tele: dict[int, float] = defaultdict(float)
        s_tot = sum(v for v in seeds.values() if v > 0)
        if s_tot > 0:
            for k, v in seeds.items():
                if v > 0:
                    tele[k] += (1 - prior_mix) * v / s_tot
        p_tot = sum(self.prior.values())
        if p_tot > 0 and prior_mix > 0:
            for k, v in self.prior.items():
                tele[k] += prior_mix * v / p_tot
        if not tele:
            return {}
        out_w = {n: sum(nb.values()) for n, nb in self.adj.items()}
        r = dict(tele)
        for _ in range(iters):
            nxt: dict[int, float] = defaultdict(float)
            dangling = 0.0
            for n, mass in r.items():
                ow = out_w.get(n, 0.0)
                if ow <= 0:
                    dangling += mass
                    continue
                for m, w in self.adj[n].items():
                    nxt[m] += damping * mass * w / ow
            for k, v in tele.items():
                nxt[k] += (1 - damping + damping * dangling) * v
            delta = sum(abs(nxt.get(k, 0.0) - r.get(k, 0.0)) for k in set(nxt) | set(r))
            r = dict(nxt)
            if delta < tol:
                break
        return r

    def neighbors(self, pid: int, k: int = 5) -> list[tuple[int, float]]:
        return sorted(self.adj.get(pid, {}).items(), key=lambda t: -t[1])[:k]


_GRAPH: ProjectGraph | None = None
_GRAPH_AT = 0.0


def reset_cache() -> None:
    global _GRAPH, _GRAPH_AT
    _GRAPH, _GRAPH_AT = None, 0.0


def load_graph(store: Store) -> ProjectGraph | None:
    """Cached graph; reloaded when project_graph.json changes. None if never built."""
    global _GRAPH, _GRAPH_AT
    p = _graph_path()
    if not p.exists():
        return None
    m = p.stat().st_mtime
    if _GRAPH is None or m != _GRAPH_AT:
        _GRAPH, _GRAPH_AT = ProjectGraph.load(store), m
    return _GRAPH
