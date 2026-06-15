"""R4 — learned/clustered fields. A non-dev corpus that keyword-inference leaves as 'general' should get
emergent, distinct fields after clustering, so field-level features stop being dead. Needs embeddings."""
import pytest

from cartograph.config import Config, db_path
from cartograph.persona.profile import build_from_corpus
from cartograph.storage import Store


def _write(root):
    # vocab the built-in keyword fields do NOT cover -> all 'general' before R4
    topics = {
        "ledger-recon": "ledger reconciliation balance statement variance close period",
        "tax-filing": "tax filing deduction bracket return liability quarterly estimate",
        "invoice-ap": "invoice vendor payable approval purchase order remittance terms",
        "budget-fp": "budget forecast variance plan actual quarter rolling projection",
        "audit-controls": "audit control evidence sample testing workpaper materiality",
        "treasury-cash": "treasury cash liquidity sweep forecast bank facility covenant",
    }
    for name, text in topics.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"n{i}.md").write_text((text + " ") * 6, encoding="utf-8")


@pytest.mark.skipif(not __import__("cartograph.embed", fromlist=["available"]).available(),
                    reason="learned fields need the semantic index (embeddings)")
def test_learned_fields_relabel_general_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    from cartograph.embed import build_index
    from cartograph.ingest import ingest_path
    from cartograph.learned_fields import learn_fields
    root = tmp_path / "corpus"
    _write(root)
    cfg = Config()
    store = Store(db_path())
    ingest_path(root, store, cfg)
    before = build_from_corpus(store).field_weights
    build_index(store, cfg)
    rep = learn_fields(store, cfg)
    assert rep["learned"] and rep["clusters"] >= 2
    after = build_from_corpus(store).field_weights
    assert len(after) > len(before)                            # R4 adds real field granularity
    assert any(f.startswith("auto:") for f in after)           # emergent fields, not keyword-derived
