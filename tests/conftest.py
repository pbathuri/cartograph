"""Test isolation: reset module-level model caches between tests so one test's trained model (under its
own CARTOGRAPH_HOME) never leaks into another's process-global cache."""
import pytest


@pytest.fixture(autouse=True)
def _reset_model_caches():
    for mod in ("cartograph.rerank_model", "cartograph.context_affinity", "cartograph.router"):
        try:
            m = __import__(mod, fromlist=["reset_cache"])
            if hasattr(m, "reset_cache"):
                m.reset_cache()
        except Exception:
            pass
    yield
