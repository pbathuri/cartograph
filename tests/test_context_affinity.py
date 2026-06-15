"""Query-contextual affinity: a project's preference is conditioned on the query CLUSTER, so the same
project can be liked in one context and disliked in another. Also covers broadened field inference."""
import json

import numpy as np
import pytest

from cartograph import context_affinity as CA


def test_field_inference_covers_non_dev_domains():
    from cartograph.ingest import infer_field
    assert infer_field("close", "reconciliation gaap journal entry accounts payable accrual") == "finance_accounting"
    assert infer_field("driver", "firmware microcontroller i2c uart interrupt gpio rtos register") == "embedded_hardware"
    assert infer_field("entry", "go-to-market market sizing value proposition stakeholder kpi") == "business_consulting"


def test_contexts_need_embeddings_or_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "feedback.jsonl").write_text("", encoding="utf-8")
    rep = CA.build_contexts(__import__("cartograph.config", fromlist=["Config"]).Config())
    assert rep["trained"] is False                     # empty log -> refuses


def test_contextual_lookup_separates_clusters(tmp_path, monkeypatch):
    """With injected query vectors (no real embedder), the SAME project gets opposite affinity in two
    different query clusters — the core property that fixes the global-affinity over-steer."""
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)
    # two clearly-separated query directions; 'payroll' is disliked in cluster A, liked in cluster B
    va, vb = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    events = []
    for _ in range(6):
        events.append({"query": "reconcile", "liked_projects": ["close"], "disliked_projects": ["payroll"]})
        events.append({"query": "form 941", "liked_projects": ["payroll"], "disliked_projects": []})
    (tmp_path / "feedback.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    # stub the embedder: 'reconcile' -> va, 'form 941' -> vb
    monkeypatch.setattr(CA, "query_vector", lambda q, cfg: va if "reconcile" in q else vb)
    rep = CA.build_contexts(__import__("cartograph.config", fromlist=["Config"]).Config(), min_events=4)
    assert rep["trained"] and rep["clusters"] == 2
    CA.reset_cache()
    model = CA.load_context_affinity()
    aff_a, _ = model.lookup(va, "payroll")             # in the 'reconcile' context
    aff_b, _ = model.lookup(vb, "payroll")             # in the 'form 941' context
    assert aff_a < 0 < aff_b                            # SAME project, opposite affinity by context
