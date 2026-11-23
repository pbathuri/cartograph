import json
import sqlite3
import time

from cartograph.config import Config, db_path
from cartograph.storage import Store


def _store():
    return Store(db_path())


def _autolog(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE summaries (id INTEGER PRIMARY KEY, startTimestamp REAL, endTimestamp REAL, "
                "appNames TEXT, summary TEXT, keyTopics TEXT, captureIds TEXT, embedding BLOB, "
                "documentPaths TEXT, browserURLs TEXT, activityType TEXT)")
    for i, (summary, topics, docs, typ, ts) in enumerate(rows, 1):
        con.execute("INSERT INTO summaries(id,startTimestamp,summary,keyTopics,documentPaths,activityType) "
                    "VALUES(?,?,?,?,?,?)", (i, ts, summary, json.dumps(topics), json.dumps(docs), typ))
    con.commit()
    con.close()


def test_ppr_spreads_to_neighbours_and_prefers_seed():
    from cartograph.graphnet import ProjectGraph
    g = ProjectGraph({1: {2: 1.0}, 2: {1: 1.0, 3: 1.0}, 3: {2: 1.0}, 4: {}}, prior={})
    r = g.ppr({1: 1.0}, damping=0.85, prior_mix=0.0)
    # closed form on the path 1-2-3 seeded at 1: r2 = 0.85 r1 / (1 - 0.85^2/2) ~ 1.331 r1, r3 = 0.425 r2
    assert abs(r[2] / r[1] - 0.85 / (1 - 0.85 ** 2 / 2)) < 1e-3
    assert abs(r[3] / r[2] - 0.425) < 1e-3
    assert r[1] > r[3] > 0
    assert r.get(4, 0.0) == 0.0
    assert abs(sum(r.values()) - 1.0) < 1e-3


def test_activity_prior_mixes_into_teleport():
    from cartograph.graphnet import ProjectGraph
    g = ProjectGraph({1: {}, 2: {}}, prior={2: 1.0})
    r = g.ppr({1: 1.0}, prior_mix=0.5)
    assert r[2] > 0 and r[1] > 0


def test_build_graph_coactivity_and_doc_mapping(tmp_path):
    from cartograph.graphnet import ProjectGraph, build_graph
    store = _store()
    a = store.upsert_project("blink", str(tmp_path / "blink"), "ml_experiment")
    b = store.upsert_project("eog_tool", str(tmp_path / "eog_tool"), "ml_experiment")
    c = store.upsert_project("cooking", str(tmp_path / "cooking"), "general")
    store.upsert_file(a, str(tmp_path / "blink/analysis/detect.py"), ".py", "1")
    store.upsert_file(b, str(tmp_path / "eog_tool/src/filters.py"), ".py", "2")
    store.upsert_file(c, str(tmp_path / "cooking/recipes.md"), ".md", "3")
    db = tmp_path / "autolog.sqlite"
    now = time.time()
    _autolog(db, [
        ("tuning blink detector thresholds", ["blink"], ["analysis/detect.py", "src/filters.py"], "coding", now),
        ("tuning blink detector again", ["blink"], ["analysis/detect.py", "src/filters.py"], "coding", now - 86400),
        ("chatting about dinner", ["food"], ["recipes.md"], "communication", now),
    ])
    out = build_graph(store, Config(), autolog_db=db, now=now)
    assert out["n_coactive"] == 1 and out["n_activity"] == 2          # communication lane dropped
    g = ProjectGraph.load(store)
    assert b in g.adj[a] and c not in g.adj.get(a, {})
    assert g.prior[a] > 0 and c not in g.prior


def test_eval_score_and_tune_accepts_only_holdout_gains():
    from cartograph.evals import EvalQuery, score, tune
    from cartograph.graphnet import ProjectGraph

    def q(i, gold, fts):
        return EvalQuery(qid=i, ts=i, text="x", gold={gold}, fts=fts, sem=[])
    # gold project 2 is ranked 2nd by text, but is the graph neighbour of the recently-active project 3
    qs = [q(i, 2, [(10 * i + 1, 1, 300), (10 * i + 2, 2, 300), (10 * i + 3, 3, 300)]) for i in range(40)]
    g = ProjectGraph({2: {3: 1.0}, 3: {2: 1.0}, 1: {}}, prior={3: 1.0, 2: 0.5})
    ctx = {"field_weights": {}, "proj_field": {}, "graph": g, "alpha": 0.0}
    base = score(qs, {"w_ppr": 0.0}, **ctx)
    boosted = score(qs, {"w_ppr": 0.8, "prior_mix": 0.5}, **ctx)
    assert base["hit@1"] == 0.0 and boosted["hit@1"] == 1.0
    res = tune(qs[:30], qs[30:], {"w_ppr": 0.0}, **ctx)
    assert res["accepted"] and res["params"]["w_ppr"] > 0
    # no gain available -> live params kept
    res2 = tune(qs[:30], qs[30:], res["params"], **ctx)
    assert not res2["accepted"] and res2["params"]["w_ppr"] == res["params"]["w_ppr"]


def test_brief_token_cost_falls_with_snippet_caps():
    from cartograph.evals import EvalQuery, score
    qs = [EvalQuery(qid=1, ts=1, text="x", gold={1},
                    fts=[(i, 1, 1200) for i in range(1, 9)], sem=[])]
    ctx = {"field_weights": {}, "proj_field": {}, "graph": None, "alpha": 0.0}
    wide = score(qs, {"snips_per_project": 3, "snippet_chars": 400}, **ctx)
    tight = score(qs, {"snips_per_project": 1, "snippet_chars": 240}, **ctx)
    assert tight["brief_tokens"] < wide["brief_tokens"] and tight["brief_recall"] == 1.0


def test_method_hints_match_cuda_and_time_series():
    from cartograph.elite.methods import match_methods
    cuda = match_methods("my CUDA kernel has low occupancy and slow shared memory reduction")
    assert cuda[0]["area"] == "cuda_kernels" and cuda[0]["tangents"]
    ts = match_methods("forecast volatility for this time series with walk forward validation")
    assert ts[0]["area"] == "time_series"
    assert match_methods("write a birthday message") == []


def test_scout_topics_filter_and_offline_cards(tmp_path, monkeypatch):
    from cartograph import scout as sc
    assert sc._clean_topic("WhatsApp chat") is None
    assert sc._clean_topic("report_2026-11-14.pdf") is None
    assert sc._clean_topic("Kernel fusion") == "Kernel fusion"
    monkeypatch.setattr(sc, "arxiv_search", lambda t, n: [{
        "source": "arxiv", "id": "2401.00001", "title": "Fused attention kernels", "summary": "warp level fusion",
        "published": "2024-01-01", "authors": ["A. Author"], "url": "https://arxiv.org/abs/2401.00001",
        "pdf": "https://arxiv.org/pdf/2401.00001"}])
    monkeypatch.setattr(sc, "github_search", lambda t, n: [])
    store = _store()
    stats = sc.scout(store, Config(), topics=["Kernel fusion"], pause=0, pdf_dir=None)
    assert stats["papers"] == 1 and not stats["errors"]
    from cartograph.ingest import ingest_path
    ingest_path(sc.frontier_dir(), store, Config())
    cards = sc.frontier_cards(store, "how do I fuse my attention kernels")
    assert cards and cards[0]["link"].startswith("https://arxiv.org/abs/")


def test_brief_carries_hints_and_respects_snippet_cap(tmp_path):
    from cartograph.ingest import ingest_path
    from cartograph.persona import build_brief
    from cartograph.persona.profile import load_persona
    d = tmp_path / "corpus" / "cuda_hw"
    d.mkdir(parents=True)
    for i in range(5):
        (d / f"k{i}.md").write_text("cuda kernel shared memory tiling occupancy notes " * 20)
    (tmp_path / "corpus" / "other").mkdir()
    (tmp_path / "corpus" / "other" / "x.md").write_text("unrelated notes")
    store = _store()
    ingest_path(tmp_path / "corpus", store, Config())
    b = build_brief("speed up my cuda kernel shared memory tiling", store, Config(), load_persona(store))
    assert b["method_hints"] and b["method_hints"][0]["area"] == "cuda_kernels"
    per = {}
    for c in b["relevant_context"]:
        per[c["project"]] = per.get(c["project"], 0) + 1
    assert max(per.values()) <= 2


def test_scout_never_sends_personal_or_codename_topics():
    from cartograph.scout import _clean_topic
    for bad in ["expected behavior of men paying on dates in Peru", "Cao Cao's Conquests", "ResumeForge",
                "iu-demo", "mobility-introduction.tsx", "NextThread"]:
        assert _clean_topic(bad) is None, bad
    for good in ["CUDA kernel fusion", "Kalman filter", "time series forecasting", "Blink detection"]:
        assert _clean_topic(good) == good
