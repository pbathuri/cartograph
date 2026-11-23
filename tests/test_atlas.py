"""The Atlas (`carto viz`): graph with entity hubs, cited context, project detail and the memory timeline,
served read-only over the real HTTP server."""
import json
import socket
import socketserver
import threading
import urllib.request

import pytest

from cartograph import memory as M
from cartograph.config import Config, db_path
from cartograph.ingest import ingest_path
from cartograph.storage import Store


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def atlas(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        d = tmp_path / name
        d.mkdir()
        (d / ".git").mkdir()
        (d / "requirements.txt").write_text("pydantic\nhttpx\npython\n")
        (d / "notes.md").write_text(f"{name}: corneal permeability of timolol, see arXiv:2401.04444\n")
    store = Store(db_path())
    for name in ("alpha", "beta", "gamma"):
        ingest_path(tmp_path / name, store, Config())
    M.remember(store, "Use Franz cells for timolol", "decision", cwd=str(tmp_path / "alpha"))
    from cartograph.viz import app
    app._ATLAS.clear()
    port = _free_port()
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), app._make_handler())
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield port, store
    httpd.shutdown()


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        body = r.read()
        return r.status, body, r.headers


def test_page_and_assets(atlas):
    port, _ = atlas
    status, html, _ = _get(port, "/")
    assert status == 200 and b"Cartograph Atlas" in html and b"unpkg.com" not in html   # works offline
    status, png, headers = _get(port, "/mascot.png")
    assert status == 200 and headers["Content-Type"] == "image/png" and png[:4] == b"\x89PNG"


def test_atlas_has_projects_entity_hubs_and_stats(atlas):
    port, _ = atlas
    d = json.loads(_get(port, "/api/atlas")[1])
    labels = {n["label"] for n in d["nodes"] if n["kind"] == "project"}
    assert {"alpha", "beta", "gamma"} <= labels
    hubs = {n["label"] for n in d["nodes"] if n["kind"] == "entity"}
    assert "pypi/pydantic" in hubs and "pypi/python" not in hubs
    assert any(e["relation"] == "PROJECT_SHARES_ENTITY" and e.get("via") for e in d["edges"])
    assert d["stats"]["entities"] > 0 and d["stats"]["memories"] == 1


def test_context_project_and_timeline(atlas):
    port, store = atlas
    ctx = json.loads(_get(port, "/api/context?q=timolol%20permeability&mode=mix")[1])
    assert ctx["chunks"] and all(c["cite"] for c in ctx["chunks"])
    assert ctx["tokens"] <= ctx["max_tokens"]
    pid = store.project_id_by_name("alpha")
    p = json.loads(_get(port, f"/api/project?id=p{pid}")[1])
    assert p["name"] == "alpha" and p["files"] == 2
    assert {r["project"] for r in p["related"]} == {"beta", "gamma"}
    assert any(e["entity"] == "package:pypi/pydantic" and e["evidence"]["cite"] for e in p["entities"])
    assert [m["text"] for m in p["memories"]] == ["Use Franz cells for timolol"]
    t = json.loads(_get(port, "/api/timeline")[1])
    assert t["memories"][0]["kind"] == "decision"


def test_atlas_cache_refreshes_when_graph_changes(atlas, tmp_path):
    port, store = atlas
    before = json.loads(_get(port, "/api/atlas")[1])["stats"]["memories"]
    M.remember(store, "Rejected: Ussing chamber", "rejected")
    after = json.loads(_get(port, "/api/atlas")[1])["stats"]["memories"]
    assert (before, after) == (1, 2)
