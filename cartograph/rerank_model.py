"""Learned reranker — a small model trained on YOUR feedback that sits between retrieval and the answer.

The heuristic blend (base rank vs field weight vs preference-vector cosine vs field-match) uses fixed
constants. This learns those weights from your `record_use`/`feedback` history: it replays each logged
query through retrieval, labels candidates by whether their project helped, and fits a logistic model on
4 interpretable features. At serve time, if a model is trained it scores candidates; otherwise the
heuristic is used unchanged. Pure NumPy (no sklearn). Honest cold-start: needs both classes + enough
examples or it refuses to train (and the heuristic keeps working).

Features per candidate (computed identically at train + serve):
  base   = 1 - rank/n          (where retrieval put it)
  fieldw = persona field weight of the chunk's project
  prefc  = preference-vector cosine in the chunk's field subspace (0 if no vector)
  fmatch = 1 if the chunk's field == the prompt's routed field else 0
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config, home

NFEAT = 5


def _path() -> Path:
    return home() / "reranker.npz"


def extract_features(base: float, field_weight: float, pref_cos: float, field_match: float,
                     proj_affinity: float = 0.0):
    import numpy as np
    return np.array([base, field_weight, max(0.0, pref_cos), field_match, proj_affinity], dtype="float32")


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def project_affinities() -> dict:
    """Per-project standing preference from the feedback log: (helped - disliked) / (helped + disliked + 1)
    in (-1, 1). Gives the reranker a PROJECT-level signal (field features can't separate same-field repos)."""
    import collections
    import json
    log = home() / "feedback.jsonl"
    if not log.exists():
        return {}
    pos, neg = collections.Counter(), collections.Counter()
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        for p in ev.get("liked_projects", []):
            pos[_norm(p)] += 1
        for p in ev.get("disliked_projects", []):
            neg[_norm(p)] += 1
    return {k: (pos[k] - neg[k]) / (pos[k] + neg[k] + 1.0) for k in set(pos) | set(neg)}


class _LogReg:
    """Minimal standardized logistic regression via gradient descent."""

    def __init__(self, w, b, mu, sd):
        self.w, self.b, self.mu, self.sd = w, b, mu, sd

    def proba(self, X):
        import numpy as np
        Z = (np.asarray(X, dtype="float32") - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    @staticmethod
    def fit(X, y, iters=400, lr=0.3, l2=1e-3):
        import numpy as np
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        w, b = np.zeros(Z.shape[1], dtype="float32"), 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = p - y
            w -= lr * (Z.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        acc = float((( (1.0 / (1.0 + np.exp(-(Z @ w + b)))) >= 0.5) == (y >= 0.5)).mean())
        return _LogReg(w, b, mu, sd), acc


def save(model: _LogReg, n: int, acc: float) -> None:
    import numpy as np
    np.savez(_path(), w=model.w, b=np.array([model.b]), mu=model.mu, sd=model.sd,
             n=np.array([n]), acc=np.array([acc]))


_MODEL = None
_LOADED = False


def load():
    global _MODEL, _LOADED
    if _LOADED:
        return _MODEL
    _LOADED = True
    if not _path().exists():
        _MODEL = None
        return None
    try:
        import numpy as np
        with np.load(_path()) as z:
            w, mu, sd = z["w"], z["mu"], z["sd"]
            if w.shape[0] != NFEAT:          # stale model from an older feature schema -> ignore (don't crash serve)
                _MODEL = None
                return None
            _MODEL = _LogReg(w, float(z["b"][0]), mu, sd)
    except Exception:
        _MODEL = None
    return _MODEL


def reset_cache() -> None:
    global _MODEL, _LOADED
    _MODEL, _LOADED = None, False


def train_from_log(store, cfg: Config, persona, *, min_examples: int = 12) -> dict:
    """Replay the feedback log through retrieval, label candidates, fit the reranker. Returns a report."""
    import json

    import numpy as np

    from .retrieve import retrieve
    from .router import route

    log = home() / "feedback.jsonl"
    if not log.exists():
        return {"trained": False, "reason": "no feedback log yet"}
    norm = _norm
    vecs = persona._load_all()
    aff = project_affinities()
    from .context_affinity import load_context_affinity, query_vector
    ctx = load_context_affinity()                          # contextual affinity (built before training)
    X, Y = [], []
    events = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        helped = {norm(p) for p in ev.get("liked_projects", [])}
        bad = {norm(p) for p in ev.get("disliked_projects", [])}
        if not (helped or bad) or not ev.get("query"):
            continue
        events += 1
        res = retrieve(ev["query"], store, cfg, top_k=12)
        chunks = res.chunks
        if not chunks:
            continue
        rfield = route(ev["query"], cfg)["field"]
        qv = query_vector(ev["query"], cfg) if ctx is not None else None
        names = list({c.get("project_name") for c in chunks if c.get("project_name")})
        ph = ",".join("?" * len(names)) if names else ""
        pf = {}
        if names:
            with store.cursor() as c:
                pf = {dict(r)["name"]: (dict(r)["field"] or "general") for r in
                      c.execute(f"SELECT name, field FROM projects WHERE name IN ({ph})", names).fetchall()}
        n = len(chunks)
        for i, ch in enumerate(chunks):
            nm = norm(ch.get("project_name"))
            label = 1.0 if nm in helped else (0.0 if nm in bad else None)
            if label is None:
                continue
            fld = pf.get(ch.get("project_name", ""), "general")
            fw = persona.field_weights.get(fld, 0.0)
            pc = 0.0
            v = vecs.get(fld) if fld in vecs else vecs.get("_global")
            if v is not None:
                from .embed import DOC_PREFIX, _model, available
                if available():
                    cv = _model(cfg.embed_model).encode([DOC_PREFIX + (ch.get("chunk_text") or "")[:600]],
                                                        normalize_embeddings=True)[0]
                    pc = float(max(0.0, np.dot(v, cv)))
            caff = ctx.lookup(qv, ch.get("project_name"))[0] if (ctx is not None and qv is not None) \
                else aff.get(nm, 0.0)                       # contextual affinity (train == serve)
            X.append(extract_features(1 - i / n, fw, pc, 1.0 if fld == rfield else 0.0, caff))
            Y.append(label)
    if len(Y) < min_examples or len(set(Y)) < 2:
        return {"trained": False, "reason": f"need >= {min_examples} labeled candidates of both classes "
                f"(have {len(Y)} from {events} events)", "examples": len(Y)}
    model, acc = _LogReg.fit(X, Y)
    save(model, len(Y), acc)
    reset_cache()
    return {"trained": True, "examples": len(Y), "events": events, "fit_accuracy": round(acc, 3),
            "weights": {n_: round(float(w_), 3) for n_, w_ in
                        zip(["base", "field_weight", "pref_cos", "field_match", "proj_affinity"], model.w)},
            "note": "fit_accuracy is on your own history (a personal recommender), not a held-out "
                    "generalization metric — proj_affinity intentionally encodes standing preferences."}
