"""Field router: keyword fallback always works; learned centroids route correctly when semantic is on."""
import pytest

from cartograph.config import Config
from cartograph.demo import write_corpus
from cartograph.ingest import ingest_path
from cartograph.router import build_centroids, route
from cartograph.storage import Store


def test_keyword_fallback_without_centroids(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    r = route("build a quant factor backtest", Config())
    assert r["method"] == "keyword"
    assert r["field"] == "quant_research"


def test_learned_router_routes_to_corpus_field(tmp_path, monkeypatch):
    from cartograph.embed import available
    if not available():
        pytest.skip("semantic extra not installed (learned router needs embeddings)")
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    import cartograph.router as R
    R._CENTS = None                                  # reset module cache for the fresh workspace
    store = Store(tmp_path / "home" / "graph.sqlite")
    ingest_path(write_corpus(tmp_path / "corpus"), store, Config())   # ml/web/data/library
    counts = build_centroids(store, Config())
    assert counts                                    # centroids trained
    R._CENTS = None
    r = route("how do I design a clean typed public API for my package", Config())
    assert r["method"] == "learned"
    assert r["field"] == "library"                   # routed to the right corpus field, no keyword needed
