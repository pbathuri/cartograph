"""The zero-setup demo corpus + the implicit-feedback MCP loop (record_use)."""
import json
import os
import subprocess
import sys
from pathlib import Path

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
    assert out and out[0]["result"]["ok"] is True
    assert out[0]["result"]["n_signals"] == 1
