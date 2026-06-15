"""End-to-end smoke: ingest the sample data -> stats -> FTS retrieve -> elite layer -> MCP.
Pure-FTS path (no ML deps needed), so it runs anywhere."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cartograph.config import Config
from cartograph.elite import elevate, frontier_report, score_build
from cartograph.ingest import ingest_path
from cartograph.retrieve import retrieve
from cartograph.storage import Store

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_data"


@pytest.fixture()
def graph(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    ingest_path(SAMPLE, store, Config())
    return store


def test_ingest_builds_graph(graph):
    s = graph.stats()
    assert s["projects"] >= 1 and s["chunks"] >= 1


def test_fts_retrieve(graph):
    res = retrieve("train model held-out metrics", graph, Config(), top_k=5)
    assert res.method == "fts"
    assert res.chunks and "chunk_text" in res.chunks[0]


def test_elite_layer(graph):
    e = elevate("build an ml model with ablations", graph, Config())
    assert e["field"] == "ml_experiment"
    assert e["elite_bar"] and e["playbook"]
    rep = frontier_report(graph, top=5)
    assert "fields" in rep
    sb = score_build(SAMPLE / "ml_project", "ml_experiment")
    assert sb["grade"] in {"minimal", "baseline", "professional", "elite"}


def test_mcp_initialize_and_tools():
    payload = (json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n" +
               json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n")
    p = subprocess.run([sys.executable, "-m", "cartograph.mcp_server"],
                       input=payload, capture_output=True, text=True, timeout=60)
    lines = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    assert lines[0]["result"]["serverInfo"]["name"] == "cartograph"
    names = {t["name"] for t in lines[1]["result"]["tools"]}
    assert {"retrieve_context", "elevate_task", "frontier_status"} <= names
