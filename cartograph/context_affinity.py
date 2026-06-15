"""Query-CONTEXTUAL project affinity — the fix for the over-steer the test_v1 trial proved.

GLOBAL affinity (one number per project) is the wrong granularity: a project can be the WRONG answer
for one kind of query and the RIGHT answer for another (e.g. 'payroll' is poor for "reconcile the
account" but correct for "Form 941 payroll"). A single global affinity therefore can't lift ambiguous
queries without risking clear ones.

This conditions affinity on the *kind of query*: we cluster the user's feedback queries by embedding,
and learn a separate project affinity PER cluster. At serve time we route the prompt to its nearest
cluster and use that cluster's affinities. So preference is contextual — payroll can be liked in the
'941' context and disliked in the 'reconcile' context simultaneously.

Needs embeddings (semantic). Without them there are no clusters, and callers fall back to global affinity
(still relevance-gated). Pure NumPy k-means; no sklearn.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config, home


def _na(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def _npz() -> Path:
    return home() / "context_affinity.npz"


def _json() -> Path:
    return home() / "context_affinity.json"


def query_vector(query: str, cfg: Config):
    """Embed a query the same way retrieval does (QUERY_PREFIX, normalized). None if no embeddings."""
    try:
        from .embed import QUERY_PREFIX, _model, available
        if not available():
            return None
        import numpy as np
        v = _model(cfg.embed_model).encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return np.asarray(v, dtype="float32")
    except Exception:
        return None


def _kmeans(X, k: int, iters: int = 30, seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        assign = (X @ C.T).argmax(1)                      # cosine (rows are unit-norm)
        newC = C.copy()
        for j in range(k):
            pts = X[assign == j]
            if len(pts):
                m = pts.mean(0)
                newC[j] = m / (np.linalg.norm(m) + 1e-9)
        if np.allclose(newC, C):
            break
        C = newC
    return C, assign


def build_contexts(cfg: Config, *, min_events: int = 8) -> dict:
    """Cluster the feedback queries and learn per-cluster project affinity. Persisted for serve + train."""
    import collections

    import numpy as np
    log = home() / "feedback.jsonl"
    if not log.exists():
        return {"trained": False, "reason": "no feedback log"}
    events = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        q = ev.get("query")
        if q and (ev.get("liked_projects") or ev.get("disliked_projects")):
            events.append(ev)
    if len(events) < min_events:
        return {"trained": False, "reason": f"need >= {min_events} feedback events (have {len(events)})"}
    vecs = [query_vector(ev["query"], cfg) for ev in events]
    if any(v is None for v in vecs):
        return {"trained": False, "reason": "embeddings unavailable (semantic extra needed)"}
    X = np.vstack(vecs).astype("float32")
    k = max(2, min(8, round(len(set(ev["query"] for ev in events)) ** 0.5)))
    k = min(k, len(X))
    C, assign = _kmeans(X, k)
    pos = [collections.Counter() for _ in range(k)]
    neg = [collections.Counter() for _ in range(k)]
    for ev, c in zip(events, assign):
        for p in ev.get("liked_projects", []):
            pos[c][_na(p)] += 1
        for p in ev.get("disliked_projects", []):
            neg[c][_na(p)] += 1
    affinities, counts = [], []
    for c in range(k):
        keys = set(pos[c]) | set(neg[c])
        affinities.append({key: (pos[c][key] - neg[c][key]) / (pos[c][key] + neg[c][key] + 1.0) for key in keys})
        counts.append({key: pos[c][key] + neg[c][key] for key in keys})
    np.savez(_npz(), centroids=C)
    _json().write_text(json.dumps({"affinities": affinities, "counts": counts, "k": k},
                                  ensure_ascii=False), encoding="utf-8")
    reset_cache()
    return {"trained": True, "clusters": k, "events": len(events),
            "cluster_sizes": [int((assign == c).sum()) for c in range(k)]}


class ContextAffinity:
    def __init__(self, centroids, affinities, counts):
        self.centroids = centroids
        self.affinities = affinities
        self.counts = counts

    def _cluster(self, qvec) -> int:
        import numpy as np
        return int(np.argmax(self.centroids @ np.asarray(qvec, dtype="float32")))

    def lookup(self, qvec, project: str) -> tuple[float, int]:
        """Return (affinity, evidence_count) for this project in the query's nearest cluster."""
        import numpy as np
        q = np.asarray(qvec, dtype="float32")
        if q.ndim != 1 or q.shape[0] != self.centroids.shape[1]:
            return (0.0, 0)                                # dim mismatch (e.g. stale model) -> neutral
        c = self._cluster(q)
        nm = _na(project)
        return float(self.affinities[c].get(nm, 0.0)), int(self.counts[c].get(nm, 0))


_MODEL = None
_LOADED = False


def reset_cache() -> None:
    global _MODEL, _LOADED
    _MODEL, _LOADED = None, False


def load_context_affinity():
    global _MODEL, _LOADED
    if _LOADED:
        return _MODEL
    _LOADED = True
    if not (_npz().exists() and _json().exists()):
        _MODEL = None
        return None
    try:
        import numpy as np
        with np.load(_npz()) as z:
            C = z["centroids"]
        d = json.loads(_json().read_text(encoding="utf-8"))
        _MODEL = ContextAffinity(C, d["affinities"], d["counts"])
    except Exception:
        _MODEL = None
    return _MODEL
