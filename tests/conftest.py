"""Test isolation: reset module-level model caches between tests so one test's trained model (under its
own CARTOGRAPH_HOME) never leaks into another's process-global cache."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Never read or write the developer's real ~/.cartograph (a real semantic index there flipped
    FTS-only tests to hybrid). Tests that need their own home still override this via monkeypatch."""
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path_factory.mktemp("carto_home")))
    yield


@pytest.fixture(autouse=True)
def _reset_model_caches():
    for mod in ("cartograph.rerank_model", "cartograph.context_affinity", "cartograph.router"):
        try:
            m = __import__(mod, fromlist=["reset_cache"])
            if hasattr(m, "reset_cache"):
                m.reset_cache()
        except Exception:
            pass
    try:
        import cartograph.retrieve as _r
        _r._INDEX = None                                   # process-global semantic index cache
    except Exception:
        pass
    yield
