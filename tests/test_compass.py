"""The compass keeps many directions at once: ambitions, parallel ventures,
temporary sorties, and threads that go quiet and come back."""
from cartograph.compass import compass_for_prompt, constellation, sync
from cartograph.storage import Store

T0 = 1_700_000_000.0
DAY = 86400.0


def _row(i, day, summary, topics, files=None, activity="coding", apps=None):
    return {
        "id": i,
        "start": T0 + day * DAY,
        "summary": summary,
        "topics": topics,
        "files": files or [],
        "activity": activity,
        "apps": apps or ["Cursor"],
    }


def test_parallel_ventures_stay_warm_together(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    rows = []
    n = 1
    for day in range(3):
        rows.append(_row(n, day, "Extended the factor notebook and wrote a new regression.",
                         ["Factor Model"], ["factor_model.ipynb"]))
        n += 1
        rows.append(_row(n, day, "Characterized the EPYC cache path in the Monte Carlo harness.",
                         ["Cache Paper"], ["cache_harness.py"]))
        n += 1
    sync(store, rows, declarations=[{
        "key": "longhorizon", "title": "PhD computational quant finance",
        "horizon": "ambition", "residue": "CMU, Stanford, or UPenn",
    }], now=T0 + 2 * DAY)
    data = constellation(store, now=T0 + 2 * DAY)
    titles = {c["title"] for c in data["warm"]}
    assert "Factor Model" in titles
    assert "Cache Paper" in titles
    assert any(a["title"].startswith("PhD") for a in data["ambitions"])
    assert data["counts"]["warm"] >= 2


def test_sortie_folds_and_keeps_residue(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    sync(store, [
        _row(1, 0, "Weekend case competition on GridCase. Built the pitch deck and the model.",
             ["GridCase"], ["pitch_deck.pptx"], activity="writing"),
    ], now=T0)
    later = constellation(store, now=T0 + 20 * DAY)
    folded = later["sorties"]["folded_residue"]
    assert len(folded) == 1
    assert "pitch" in folded[0]["residue"].lower() or "case" in folded[0]["residue"].lower()
    assert all(c["title"] != "GridCase" for c in later["warm"])


def test_quiet_thread_echoes_when_it_returns(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    sync(store, [
        _row(1, 0, "First pass on the ocular segmentation notebook.", ["Ocular"], ["ocular.ipynb"]),
        _row(2, 1, "Labeled another recording for ocular segmentation.", ["Ocular"], ["ocular.ipynb"]),
    ], now=T0 + DAY)
    resting = constellation(store, now=T0 + 40 * DAY)
    assert any(c["title"] == "Ocular" for c in resting["resting"])
    sync(store, [
        _row(3, 40, "Came back to ocular segmentation and fixed the blink window.", ["Ocular"], ["ocular.ipynb"]),
    ], now=T0 + 40 * DAY)
    data = constellation(store, now=T0 + 40 * DAY)
    echoes = [c for c in data["echoes"] if c["title"] == "Ocular"]
    assert echoes and echoes[0]["echoes"] >= 1
    assert echoes[0]["residue"]


def test_life_lane_does_not_become_a_direction(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    report = sync(store, [
        _row(1, 0, "Watching an episode of a show on Arc.", ["Arc"], activity="watching", apps=["Arc"]),
    ], now=T0)
    data = constellation(store, now=T0)
    assert data["counts"]["directions"] == 0
    assert report["life_skipped"] == 1


def test_prompt_reopens_a_resting_thread_without_waking_the_others(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    rows = []
    n = 1
    for day in range(3):
        rows.append(_row(n, day, "Working the factor model regression.", ["Factor Model"], ["factor.py"]))
        n += 1
    rows.append(_row(n, 0, "Drafted the cache paper methods.", ["Cache Paper"], ["methods.tex"]))
    rows.append(_row(n + 1, 1, "Revised the cache paper methods.", ["Cache Paper"], ["methods.tex"]))
    sync(store, rows, now=T0 + 2 * DAY)
    # Factor is still warm; cache has gone quiet.
    data = compass_for_prompt(store, "extend the cache paper methods section", now=T0 + 40 * DAY)
    names = [c["title"] for c in data["relevant"]]
    assert "Cache Paper" in names
    warm = {c["title"] for c in data["warm"]}
    assert "Cache Paper" not in warm


def test_sync_is_incremental(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    sync(store, [_row(1, 0, "Started the factor model.", ["Factor Model"], ["factor.py"]),
                 _row(2, 1, "Continued the factor model.", ["Factor Model"], ["factor.py"])],
         now=T0 + DAY)
    again = sync(store, [_row(1, 0, "Started the factor model.", ["Factor Model"], ["factor.py"]),
                         _row(3, 2, "Added a figure to the factor model.", ["Factor Model"], ["figure.py"])],
                 now=T0 + 2 * DAY)
    data = constellation(store, now=T0 + 2 * DAY)
    factor = next(c for c in data["warm"] if c["title"] == "Factor Model")
    assert factor["touch_days"] == 3
    assert again["observations_linked"] == 1


def test_stamped_topic_that_the_summary_never_says_is_ignored(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    sync(store, [
        _row(1, 0, "Solving a divided differences table for the interpolation homework.",
             ["Pigbet", "AutoResearchClaw", "Divided Differences"], ["hw.py"]),
        _row(2, 1, "Finished the divided differences table and checked the error bound.",
             ["Pigbet", "Divided Differences"], ["hw.py"]),
    ], now=T0 + DAY)
    data = constellation(store, now=T0 + DAY)
    titles = {c["title"] for c in data["warm"] + data["sparks"]}
    assert "Divided Differences" in titles
    assert "Pigbet" not in titles
    assert "AutoResearchClaw" not in titles


def test_declared_plan_is_not_pretending_to_be_this_week(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    sync(store, [], declarations=[{
        "key": "capstone", "title": "Capstone accelerated finance",
        "horizon": "venture", "residue": "starts 2027-08",
    }], now=T0)
    data = constellation(store, now=T0)
    assert any(c["title"].startswith("Capstone") for c in data["held"])
    assert all(not c["title"].startswith("Capstone") for c in data["warm"])
