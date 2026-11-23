import pytest

from cartograph.config import Config, db_path, index_dir
from cartograph.storage import Store


def test_index_is_incremental_and_swaps_atomically(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    import numpy as np

    from cartograph.embed import build_index
    cfg, store = Config(), Store(db_path())
    pid = store.upsert_project("p", str(tmp_path / "p"), "ml_experiment")
    f1 = store.upsert_file(pid, "a.md", ".md", "h1")
    store.add_chunks(f1, ["gradient descent notes", "walk forward backtest", "roofline model"])
    first = build_index(store, cfg)
    assert first["embedded"] == 3 and first["reused"] == 0
    v_before = np.load(index_dir() / "vectors.npy")[:3].copy()

    f2 = store.upsert_file(pid, "b.md", ".md", "h2")
    store.add_chunks(f2, ["deflated sharpe ratio"])
    second = build_index(store, cfg)
    assert second == {**second, "vectors": 4, "embedded": 1, "reused": 3}
    assert np.array_equal(np.load(index_dir() / "vectors.npy")[:3], v_before)   # reused rows copied exactly
    assert not (index_dir().parent / "index.building").exists()

    assert build_index(store, cfg, full=True)["embedded"] == 4
