"""R4 — learned/clustered fields: give field-level features a substrate for NON-dev users.

Keyword field-inference (ingest.py) is dev-centric; a finance/legal/clinical corpus collapses to
`general`, which kills field weights + per-field preference subspaces for exactly the users who can't
declare fields. This learns emergent fields by clustering the corpus's own embeddings and relabels the
`general` projects with their cluster — so field-level signals work for ANY vocabulary, no keyword list.

Reuses the already-built semantic index (no re-embedding). Only touches projects left as `general`
(declared/keyword-inferred fields are kept). No-op without the semantic index. Pure-NumPy k-means.
"""
from __future__ import annotations

import collections
import re

from .config import Config
from .storage import Store

_GENERIC = {"the", "and", "for", "with", "note", "general", "data", "project", "main", "test", "src",
            "app", "lib", "core", "util", "utils", "common", "new", "old", "tmp", "v1", "v2"}
_WORD = re.compile(r"[a-z][a-z0-9]{2,}")


def _project_vectors(store: Store, cfg: Config):
    """Mean (unit-norm) embedding per project id, from the existing index. None if no index."""
    import numpy as np
    from .embed import SemanticIndex
    if not SemanticIndex.exists():
        return None, None
    idx = SemanticIndex(cfg)
    vecs = np.asarray(idx.vectors, dtype="float32")
    pids = np.asarray(idx.project_ids)
    by_proj: dict[int, list] = collections.defaultdict(list)
    for v, p in zip(vecs, pids):
        if p != -1:
            by_proj[int(p)].append(v)
    means, order = [], []
    for pid, vs in by_proj.items():
        m = np.mean(vs, axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            means.append(m / n)
            order.append(pid)
    if not means:
        return None, None
    return np.vstack(means).astype("float32"), order


def _name_cluster(names: list[str]) -> str:
    """A human-meaningful field name from the cluster's project names (most distinctive shared token)."""
    toks = collections.Counter()
    for nm in names:
        for t in _WORD.findall(nm.lower().replace("-", " ").replace("_", " ")):
            if t not in _GENERIC:
                toks[t] += 1
    if toks:
        return "auto:" + toks.most_common(1)[0][0]
    return "auto:cluster"


def learn_fields(store: Store, cfg: Config, *, only_general: bool = True, min_projects: int = 4) -> dict:
    """Cluster project embeddings and relabel `general` projects with their learned field. Returns a report."""
    from .context_affinity import _kmeans
    means, order = _project_vectors(store, cfg)
    if means is None:
        return {"learned": False, "reason": "no semantic index (run carto index)"}
    with store.cursor() as c:
        fields = {int(dict(r)["id"]): (dict(r)["field"] or "general")
                  for r in c.execute("SELECT id, field FROM projects").fetchall()}
        names = {int(dict(r)["id"]): dict(r)["name"]
                 for r in c.execute("SELECT id, name FROM projects").fetchall()}
    targets = [i for i, pid in enumerate(order)
               if (fields.get(pid, "general") == "general" or not only_general)]
    if len(targets) < min_projects:
        return {"learned": False, "reason": f"only {len(targets)} unfielded projects (< {min_projects})"}
    X = means[targets]
    k = max(2, min(8, round(len(targets) ** 0.5)))
    k = min(k, len(X))
    _, assign = _kmeans(X, k)
    clusters: dict[int, list[int]] = collections.defaultdict(list)  # cluster -> [project_id]
    for t, c_ in zip(targets, assign):
        clusters[int(c_)].append(order[t])
    assigned = {}
    used: dict[str, int] = {}
    with store.cursor() as cur:
        for cl, pids in clusters.items():
            label = _name_cluster([names[p] for p in pids])
            if label in used:                              # disambiguate duplicate names
                used[label] += 1
                label = f"{label}{used[label]}"
            else:
                used[label] = 1
            for pid in pids:
                cur.execute("UPDATE projects SET field=? WHERE id=?", (label, pid))
                assigned[names[pid]] = label
    return {"learned": True, "clusters": len(clusters), "projects": len(targets), "fields": assigned}
