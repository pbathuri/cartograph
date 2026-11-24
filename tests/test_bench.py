"""`carto bench`: metric functions, the bundled citation corpus, and phrase-aware keyword queries."""
import json

import pytest

from cartograph import bench
from cartograph.config import home
from cartograph.documents import extract
from cartograph.retrieve import sanitize


def test_metrics_match_hand_computed_values():
    rel = {"a": 1, "b": 1}
    assert bench.ndcg_at(["a", "b", "c"], rel) == pytest.approx(1.0)
    assert bench.ndcg_at(["c", "a"], {"a": 1}) == pytest.approx(1 / 1.5849625, rel=1e-6)
    assert bench.ndcg_at(["x"], {}) == 0.0
    assert bench.recall_at(["a", "c"], rel, 10) == 0.5
    assert bench.mrr_at(["c", "d", "b"], rel) == pytest.approx(1 / 3)
    assert bench.mrr_at(["c"], rel) == 0.0
    lat = bench._lat([5.0, 1.0, 3.0])
    assert lat["p50_ms"] == 3.0 and lat["p95_ms"] == 5.0


def test_pdf_writer_round_trips_through_extraction(tmp_path):
    pytest.importorskip("pypdf")
    p = tmp_path / "two.pdf"
    p.write_bytes(bench.pdf_bytes(["Alpha page text", "Beta (parenthesised) text"]))
    secs = extract(p)
    assert [s.locator for s in secs] == ["p.1", "p.2"]
    assert "Alpha" in secs[0].text and "parenthesised" in secs[1].text


def test_citation_corpus_is_deterministic(tmp_path):
    a = bench.make_citation_corpus(tmp_path / "a", n_facts=30)
    b = bench.make_citation_corpus(tmp_path / "b", n_facts=30)
    assert a == b and len(a) == 30
    kinds = {t["file"].rsplit(".", 1)[1] for t in a}
    assert {"pdf", "docx", "pptx", "md", "py"} <= kinds


def test_citation_suite_finds_and_cites(monkeypatch):
    pytest.importorskip("pypdf")
    out = bench.run_citations(n_facts=60)
    assert out["queries"] == 60
    assert out["hit@5"] >= 0.95
    assert out["citation@1"] >= 0.9
    assert set(out["citation@1_by_type"]) == {"docx", "md", "pdf", "pptx", "py"}


def test_run_merges_suites_and_renders_markdown(monkeypatch):
    fake = {"suite": "citations", "queries": 1, "documents": 1, "chunks": 1, "hit@5": 1.0,
            "citation@1": 1.0, "citation@1_by_type": {"md": 1.0}, "p50_ms": 1.0, "p95_ms": 1.0}
    sci = {"suite": "scifact", "queries": 2, "documents": 3,
           "fts": {"ndcg@10": 0.5, "recall@100": 0.9, "mrr@10": 0.4, "p50_ms": 2.0, "p95_ms": 3.0}}
    monkeypatch.setattr(bench, "run_citations", lambda **k: fake)
    monkeypatch.setattr(bench, "run_scifact", lambda **k: sci)
    bench.run(["scifact"])
    rep = bench.run(["citations"])
    assert [r["suite"] for r in rep["results"]] == ["scifact", "citations"]
    assert json.loads((home() / "bench" / f"results-{rep['date']}.json").read_text())["results"] == rep["results"]
    md = bench.markdown(rep)
    assert "nDCG@10 **0.500**" in md and "citation@1 **1.000**" in md
    with pytest.raises(ValueError):
        bench.run(["nope"])


def test_sanitize_keeps_identifiers_as_phrases():
    q = sanitize("residence time of KX-417 vs torch.nn")
    assert '"KX 417"' in q and '"torch nn"' in q and '"residence"' in q
    assert sanitize("plain words only") == '"plain" OR "words"'
    assert sanitize("how do I size positions with volatility") == '"size" OR "positions" OR "volatility"'
    assert sanitize("what is it") == '"what" OR "is" OR "it"'      # all stopwords: keep them
