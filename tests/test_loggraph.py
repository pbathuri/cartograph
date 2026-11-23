"""Consecutive logs carry claims forward and link by bubble-sorted similarity."""
from cartograph.compass import sync
from cartograph.loggraph import assemble, bubble_rank, infer_record, ingest_logs
from cartograph.storage import Store

T0 = 1_700_000_000.0
DAY = 86400.0


def test_bubble_sort_puts_the_strongest_similarity_first():
    ranked = bubble_rank([(0.1, "a"), (0.9, "b"), (0.4, "c"), (0.7, "d")])
    assert [item for _, item in ranked] == ["b", "d", "c", "a"]


def test_unanswered_question_is_carried_into_the_next_log():
    first = infer_record("How should the blink window be estimated for this recording?")
    second = infer_record("Adjusted the sampling rate for the recording.", first)
    assert any("blink window" in q for q in second["carried"])
    assert second["answered"] == []


def test_delta_does_not_repeat_a_claim_the_previous_log_already_holds():
    first = infer_record("Extended the factor notebook and wrote a new regression.")
    second = infer_record(
        "Extended the factor notebook and wrote a new regression. Added a residual plot.",
        first,
    )
    delta = " ".join(second["delta"]).lower()
    assert "residual" in delta
    assert "regression" not in delta


def test_chain_walks_forward_and_similar_neighbor_outranks_unrelated(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    ingest_logs(store, [
        {"id": 1, "start": 1, "summary":
         "Extended the factor model regression in the notebook and checked the residuals carefully."},
        {"id": 2, "start": 2, "summary":
         "Continued the factor model regression and plotted the residuals in the notebook again."},
        {"id": 3, "start": 3, "summary":
         "Watched a cooking video about pasta sauce and olive oil on the television."},
    ])
    data = assemble(store, "factor model regression residuals")
    prose = " ".join(node["prose"] for node in data["chain"])
    assert "factor model" in prose
    assert "pasta" not in prose
    assert data["similar"]
    assert all("pasta" not in node["prose"] for node in data["similar"])
    with store.cursor() as c:
        top = c.execute(
            "SELECT src, dst, score FROM log_edges WHERE relation='SIMILAR' ORDER BY score DESC LIMIT 1"
        ).fetchone()
    assert top is not None
    assert top["score"] > 0.22
    assert {top["src"], top["dst"]} == {1, 2}


def test_unrelated_clock_neighbor_does_not_become_the_parent(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    ingest_logs(store, [
        {"id": 1, "start": 1, "summary":
         "How should the blink window be estimated for this ocular recording?"},
        {"id": 2, "start": 2, "summary":
         "Watched a cooking video about pasta sauce and olive oil on the television."},
        {"id": 3, "start": 3, "summary":
         "Adjusted the sampling rate and estimated the blink window for the ocular recording."},
    ])
    data = assemble(store, "blink window ocular recording")
    prose = " ".join(node["prose"] for node in data["chain"])
    assert "blink window" in prose
    assert "pasta" not in prose
    with store.cursor() as c:
        carried = c.execute(
            "SELECT carried FROM log_nodes WHERE summary_id=3"
        ).fetchone()["carried"]
    assert "pasta" not in carried


def test_life_log_is_not_linked_into_the_reasoning_chain(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    sync(store, [
        {"id": 1, "start": T0, "summary": "Extended the factor model regression and checked residuals.",
         "topics": ["Factor Model"], "files": ["factor.py"], "activity": "coding", "apps": ["Cursor"]},
        {"id": 2, "start": T0 + DAY, "summary": "Watching an episode of a show on Arc.",
         "topics": ["Arc"], "files": [], "activity": "watching", "apps": ["Arc"]},
        {"id": 3, "start": T0 + 2 * DAY, "summary": "Continued the factor model regression and plotted residuals.",
         "topics": ["Factor Model"], "files": ["factor.py"], "activity": "coding", "apps": ["Cursor"]},
    ], now=T0 + 2 * DAY)
    data = assemble(store, "factor model regression")
    prose = " ".join(node["prose"] for node in data["chain"])
    assert "episode" not in prose
    assert "factor model" in prose
