"""Regression test for the reranker over-steer found in the test_v1 accountant trial.

A learned reranker with a strong GLOBAL project affinity must NOT drag a favored-but-irrelevant project
above the genuinely-relevant top hit (that bug regressed clear-query MRR 1.0 -> 0.44). The relevance gate
+ top-hit demotion protection must keep the relevant project on top.
"""
from cartograph.config import Config, db_path
from cartograph.persona.profile import load_persona
from cartograph.persona.signals import record_feedback
from cartograph.persona.steer import personalized_retrieve
from cartograph.rerank_model import reset_cache, train_from_log
from cartograph.storage import Store


def _corpus(root):
    # disjoint topic words + a shared weak term ('review process') so a query can surface BOTH.
    (root / "widgets").mkdir(parents=True)
    (root / "widgets" / "calibrate.md").write_text(
        "widget calibration torque spindle alignment gauge tolerance micrometer review process",
        encoding="utf-8")
    (root / "ledger").mkdir(parents=True)
    (root / "ledger" / "books.md").write_text(
        "invoice ledger accounts payable reconciliation vendor remittance review process", encoding="utf-8")


def test_high_affinity_project_cannot_displace_relevant_top_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    from cartograph.ingest import ingest_path
    root = tmp_path / "corpus"
    _corpus(root)
    cfg = Config()
    store = Store(db_path())
    ingest_path(root, store, cfg)

    # Hammer a global preference for 'ledger' (and dislike 'widgets') on a query that retrieves BOTH, so
    # the reranker trains on both classes and learns a dominant ledger affinity / negative widgets one.
    persona = load_persona(store)
    for _ in range(8):
        persona = record_feedback(persona, store, cfg, query="ledger reconciliation review process",
                                  liked_projects=["ledger"], disliked_projects=["widgets"])
    rep = train_from_log(store, cfg, persona)
    assert rep["trained"], rep                      # reranker trained with a dominant ledger affinity
    reset_cache()

    # A query squarely about widgets (with the shared 'review' term so ledger is also retrieved as a
    # high-affinity distractor) must STILL surface 'widgets' first: the disliked-but-relevant top hit is
    # protected, and the favored-but-less-relevant 'ledger' cannot leapfrog it.
    res = personalized_retrieve("widget calibration torque gauge tolerance review", store, cfg, persona,
                                top_k=5)
    assert res["reranker"] is True                  # the learned reranker IS active...
    assert "ledger" in res["projects"]              # ...the high-affinity distractor IS in the candidate set...
    assert res["projects"][0] == "widgets"          # ...yet relevance still wins (no over-steer)
