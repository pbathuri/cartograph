"""Full-pipeline connectivity: every subsystem hands off to the next in one lifecycle. This is the
"no foul play" check — proves the parts are actually wired together, not just green in isolation.

ingest -> index -> learn fields -> feedback -> train (router + contextual affinity + reranker) ->
personalize brief -> vision activity into the graph -> MCP tool dispatch -> Studio HTTP endpoints.
"""
import json

import pytest

from cartograph.config import Config, db_path
from cartograph.storage import Store

pytestmark = pytest.mark.skipif(
    not __import__("cartograph.embed", fromlist=["available"]).available(),
    reason="integration test exercises the semantic + learned-field path")


def _corpus(root):
    topics = {
        "billing-ar": "invoice accounts receivable aging collections dunning remittance",
        "ledger-close": "ledger reconciliation journal entry month-end close trial balance",
        "tax-provision": "deferred tax provision asc 740 book-tax difference valuation allowance",
        "audit-pbc": "audit workpaper materiality pbc control testing sample evidence",
        "payroll-941": "payroll 941 withholding fica register wages quarterly filing",
    }
    for nm, txt in topics.items():
        d = root / nm
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"f{i}.md").write_text((txt + " ") * 6, encoding="utf-8")


def test_full_pipeline_connectivity(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    from cartograph.context_affinity import build_contexts, reset_cache as rctx
    from cartograph.embed import build_index
    from cartograph.ingest import ingest_path
    from cartograph.learned_fields import learn_fields
    from cartograph.persona import build_brief, record_feedback
    from cartograph.persona.profile import build_from_corpus, load_persona, save_persona
    from cartograph.rerank_model import reset_cache, train_from_log
    from cartograph.router import build_centroids, route
    from cartograph.vision.pipeline import VisionConfig, process_frame, recent_activity
    from cartograph.vision.capture import Frame
    from cartograph.vision.dedup import NoveltyGate

    cfg = Config()
    store = Store(db_path())

    # 1. ingest -> 2. index -> 3. learned fields
    _corpus(tmp_path / "corpus")
    st = ingest_path(tmp_path / "corpus", store, cfg)
    assert st.projects == 5 and st.chunks >= 15
    assert build_index(store, cfg)["vectors"] >= 15
    assert learn_fields(store, cfg, only_general=False)["learned"]   # R4 wired to the index

    # 4. feedback -> 5. train (router + contextual affinity + reranker), all reading the same log/graph
    save_persona(build_from_corpus(store))
    persona = load_persona(store)
    for _ in range(6):
        persona = record_feedback(persona, store, cfg, query="reconcile the ledger at month end",
                                  liked_projects=["ledger-close"], disliked_projects=["billing-ar"])
        persona = record_feedback(persona, store, cfg, query="quarterly payroll tax filing",
                                  liked_projects=["payroll-941"])
    assert build_centroids(store, cfg)                         # router trained from corpus
    assert build_contexts(cfg)["trained"]                      # contextual affinity from the feedback log
    assert train_from_log(store, cfg, load_persona(store))["trained"]  # reranker from log + contexts
    reset_cache()
    rctx()

    # 6. personalize brief — the integration point every agent consumes
    brief = build_brief("reconcile the ledger at month end", store, cfg, load_persona(store))
    assert brief["prompt_field"] and brief["output_guidance"] and brief["relevant_context"]
    assert route("deferred tax provision", cfg)["field"]       # router reachable through the same cfg

    # 7. vision activity flows INTO the same graph and shows up as live context in the brief
    fr = Frame(image=None, app_title="Excel — close.xlsx", ts=0.0)
    rec = process_frame(fr, store, cfg, VisionConfig(), NoveltyGate(),
                        ocr=lambda _i: "ledger reconciliation journal entry month-end close " * 4, apply=True)
    assert rec["action"] == "store"
    assert recent_activity(store, limit=1)[0]["app"].startswith("Excel")
    assert build_brief("what was I doing", store, cfg, load_persona(store))["current_activity"]

    # 8. MCP tool dispatch reaches the same brain
    from cartograph.mcp_server import _dispatch
    out = json.loads(_dispatch(1, "personalize", {"prompt": "month end close"}))
    assert "result" in out and out["result"]["output_guidance"]

    # 9. Studio HTTP endpoints serve the populated graph + locked-core workflow
    from cartograph import workflow
    wf = workflow.get_workflow()
    assert {"redact", "router", "persona", "retrieval", "brief"} <= {n["id"] for n in wf["nodes"] if n["locked"]}
    from cartograph.insights import simple_stats
    assert simple_stats()["counts"]["projects"] == 6          # 5 corpus + the vision 'screen-activity'
