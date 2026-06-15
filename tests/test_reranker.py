"""Learned reranker: cold-start fallback (no model), and train-from-log activates + reorders."""
import numpy as np

from cartograph import rerank_model as RM
from cartograph.rerank_model import _LogReg, extract_features


def test_cold_start_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    RM.reset_cache()
    assert RM.load() is None                      # nothing trained -> heuristic path is used


def test_logreg_learns_separable_signal():
    # candidates where field_match (feature idx 3) perfectly predicts 'helped'
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(60):
        fmatch = rng.integers(0, 2)
        X.append(extract_features(rng.random(), rng.random(), rng.random(), float(fmatch)))
        y.append(float(fmatch))
    model, acc = _LogReg.fit(X, y)
    assert acc >= 0.9                             # learns the predictive feature
    # a field-matched candidate scores higher than a non-matched one, all else equal
    hi = model.proba([extract_features(0.5, 0.5, 0.5, 1.0)])[0]
    lo = model.proba([extract_features(0.5, 0.5, 0.5, 0.0)])[0]
    assert hi > lo


def test_project_affinities_from_log(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True)
    log = tmp_path / "home" / "feedback.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in [
        {"liked_projects": ["good-repo"], "disliked_projects": ["bad-repo"]},
        {"liked_projects": ["good-repo"], "disliked_projects": ["bad-repo"]},
    ]), encoding="utf-8")
    aff = RM.project_affinities()
    assert aff["goodrepo"] > 0 and aff["badrepo"] < 0      # standing preference signal, signed


def test_stale_model_dim_mismatch_ignored(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True)
    np.savez(RM._path(), w=np.zeros(4), b=np.array([0.0]), mu=np.zeros(4), sd=np.ones(4))  # 4-feat (stale)
    RM.reset_cache()
    assert RM.load() is None                               # wrong feature dim -> ignored, never crashes serve


def test_train_from_log_needs_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    from cartograph.config import Config
    from cartograph.demo import write_corpus
    from cartograph.ingest import ingest_path
    from cartograph.persona.profile import load_persona
    from cartograph.storage import Store
    store = Store(tmp_path / "home" / "graph.sqlite")
    ingest_path(write_corpus(tmp_path / "corpus"), store, Config())
    rep = RM.train_from_log(store, Config(), load_persona(store))   # no feedback log yet
    assert rep["trained"] is False                # honest cold-start: refuses without data
