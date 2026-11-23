"""The zero-setup demo corpus + the implicit-feedback MCP loop (record_use)."""
import json
import os
import subprocess
import sys

from cartograph.config import Config
from cartograph.demo import write_corpus
from cartograph.ingest import ingest_path
from cartograph.storage import Store


def test_demo_corpus_is_multifield(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    store = Store(tmp_path / "g.sqlite")
    st = ingest_path(corpus, store, Config())
    assert st.projects == 4
    # the synthetic projects should auto-detect distinct fields (no focus declared)
    assert {"ml_experiment", "web_frontend", "data_pipeline", "library"} <= set(st.fields)


def test_carto_demo_shows_citations_graph_and_memory():
    from rich.console import Console

    from cartograph.demo import run_demo
    before = os.environ.get("CARTOGRAPH_HOME")
    con = Console(record=True, width=140)
    run_demo(con)
    out = con.export_text()
    assert "chapter3.docx § Results" in out and "defense.pptx slide 3" in out
    assert "retina-seg ⇄ thesis" in out and "arxiv:1505.04597" in out
    assert "Cartograph memory for retina-seg" in out and "possibly stale" in out
    assert os.environ.get("CARTOGRAPH_HOME") == before


def test_mcp_record_use_closes_loop(tmp_path):
    home = tmp_path / "home"
    store = Store(home / "graph.sqlite")
    ingest_path(write_corpus(tmp_path / "corpus"), store, Config())
    env = {**os.environ, "CARTOGRAPH_HOME": str(home)}
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "record_use",
                                     "arguments": {"query": "api", "helped": ["typed-sdk"]}}}) + "\n"
    p = subprocess.run([sys.executable, "-m", "cartograph.mcp_server"],
                       input=payload, capture_output=True, text=True, timeout=60, env=env)
    out = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    res = json.loads(out[0]["result"]["content"][0]["text"])   # MCP-spec content-wrapped result
    assert res["ok"] is True
    assert res["n_signals"] == 1
