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
