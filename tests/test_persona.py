"""Persona layer: build from corpus, feedback adapts field weights, personalized re-rank, steering brief.
Pure-FTS / no-ML path (preference VECTOR is exercised only when semantic is installed)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cartograph.config import Config
from cartograph.ingest import ingest_path
from cartograph.persona import build_brief, record_feedback
from cartograph.persona.profile import build_from_corpus, load_persona, save_persona
from cartograph.persona.steer import personalized_retrieve
from cartograph.storage import Store

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_data"


@pytest.fixture()
def graph(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    store = Store(tmp_path / "home" / "graph.sqlite")
    # declare focus so the sample projects get real field labels
    ingest_path(SAMPLE, store, Config(field_focus=["ml_experiment"]))
    return store


def test_persona_builds_from_corpus(graph):
    p = build_from_corpus(graph)
    assert p.field_weights                      # learned something from the corpus
    assert abs(sum(p.field_weights.values()) - 1.0) < 1e-6   # normalized


def test_feedback_adapts_field_weights(graph):
    p = load_persona(graph)
    before = p.field_weights.get("ml_experiment", 0.0)
    p = record_feedback(p, graph, Config(), query="train", liked_projects=["ml_project"], weight=2.0)
    assert p.n_signals == 1
    assert p.field_weights.get("ml_experiment", 0.0) >= before   # nudged up (or held if already dominant)
    # persisted
    assert load_persona(graph).n_signals == 1


def test_personalized_retrieve_runs(graph):
    p = load_persona(graph)
    r = personalized_retrieve("train a model with held-out metrics", graph, Config(), p, top_k=5)
    assert r["personalized"] is True
    assert "chunks" in r


def test_build_brief_shape(graph):
    p = load_persona(graph)
    b = build_brief("how do I train a model", graph, Config(), p)
    assert b["prompt_field"] == "ml_experiment"
    assert b["output_guidance"] and isinstance(b["output_guidance"], list)
    assert "persona_summary" in b


def test_bidirectional_feedback_moves_weight_down(graph):
    from cartograph.persona.profile import PersonaProfile, save_persona
    # ml_project ingests as ml_experiment (declared focus); disliking it pulls ml_experiment down.
    p = PersonaProfile(field_weights={"ml_experiment": 0.6, "web_frontend": 0.4})
    save_persona(p)
    before = p.field_weights["ml_experiment"]
    p2 = record_feedback(p, graph, Config(), query="x", disliked_projects=["ml_project"], weight=1.0)
    assert p2.field_weights["ml_experiment"] < before


def test_prefs_flow_into_brief(graph):
    p = load_persona(graph)
    p.preferences = {"verbosity": "concise", "format": "bullets"}
    from cartograph.persona.profile import save_persona
    save_persona(p)
    b = build_brief("how do I train a model", graph, Config(), load_persona(graph))
    assert b["preferences"]["verbosity"] == "concise"
    assert any("verbosity=concise" in g for g in b["output_guidance"])


def test_per_field_subspaces_form(tmp_path, monkeypatch):
    """Liking chunks in different fields should create distinct per-field preference vectors."""
    from cartograph.demo import write_corpus
    from cartograph.embed import available
    if not available():
        pytest.skip("semantic extra not installed (subspace vectors require embeddings)")
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    store = Store(tmp_path / "home" / "graph.sqlite")
    ingest_path(write_corpus(tmp_path / "corpus"), store, Config())
    with store.cursor() as c:
        def cid(field):
            r = c.execute("SELECT ch.id FROM chunks ch JOIN files f ON f.id=ch.file_id "
                          "JOIN projects p ON p.id=f.project_id WHERE p.field=? LIMIT 1", (field,)).fetchone()
            return r[0] if r else None
        ml, web = cid("ml_experiment"), cid("web_frontend")
    p = load_persona(store)
    if ml:
        p = record_feedback(p, store, Config(), liked_chunks=[ml])
    if web:
        p = record_feedback(p, store, Config(), liked_chunks=[web])
    keys = set(p._load_all().keys())
    assert "_global" in keys
    assert {"ml_experiment", "web_frontend"} & keys      # at least one distinct subspace formed


def test_learned_alpha_rises_on_predicted_hits_falls_on_misses(graph):
    from cartograph.persona.profile import PersonaProfile, save_persona
    # persona already emphasizes ml_experiment; liking an ml project = predicted hit -> alpha up
    p = PersonaProfile(field_weights={"ml_experiment": 0.9, "web_frontend": 0.1}, learned_alpha=0.35)
    save_persona(p)
    p = record_feedback(p, graph, Config(), liked_projects=["ml_project"])  # ml_project -> ml_experiment
    assert p.learned_alpha > 0.35                      # persona predicted the useful field -> trust more
    # now ml_experiment is OUTSIDE the top-3 -> liking an ml project is a 'miss' -> alpha comes down
    p2 = PersonaProfile(field_weights={"hpc": 0.4, "quant": 0.3, "web_frontend": 0.2, "ml_experiment": 0.1},
                        learned_alpha=0.5)
    save_persona(p2)
    p2 = record_feedback(p2, graph, Config(), liked_projects=["ml_project"])  # ml not in top-3 -> miss
    assert p2.learned_alpha < 0.5


def test_mcp_personalize_tool(graph, monkeypatch):
    # the persona MCP tool returns a brief; run the server as a subprocess against this workspace
    env_home = str(Path(graph.path).parent)
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "personalize", "arguments": {"prompt": "train a model"}}}) + "\n"
    import os
    env = {**os.environ, "CARTOGRAPH_HOME": env_home}
    p = subprocess.run([sys.executable, "-m", "cartograph.mcp_server"],
                       input=payload, capture_output=True, text=True, timeout=60, env=env)
    out = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    assert out and "result" in out[0], p.stdout + p.stderr
    assert "output_guidance" in out[0]["result"]
