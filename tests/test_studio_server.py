"""End-to-end on the real HTTP server: studio page serves with token injected, reads work, mutations
require the token, locked deletes return 423, and a disallowed Origin is never echoed back."""
import json
import socket
import threading
import urllib.request

import pytest


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))
    from cartograph.config import db_path
    from cartograph.storage import Store
    Store(db_path()).upsert_project("p", "/p", "library")     # so launch() sees a graph
    import socketserver

    from cartograph.security import api_token
    from cartograph.viz.app import _make_handler
    port = _free_port()
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), _make_handler())
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield port, api_token()
    httpd.shutdown()


def _get(port, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    r = urllib.request.urlopen(req, timeout=5)
    return r.status, r.read().decode(), r.headers


def _post(port, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req, timeout=5)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_studio_page_injects_token(server):
    port, tok = server
    status, html, _ = _get(port, "/studio")
    assert status == 200 and "__CARTO_TOKEN__" not in html and tok in html


def test_workflow_read_ok(server):
    port, _ = server
    status, body, _ = _get(port, "/api/workflow")
    assert status == 200 and any(n["id"] == "persona" for n in json.loads(body)["nodes"])


def test_mutation_requires_token(server):
    port, tok = server
    code, _ = _post(port, "/api/workflow/add", {"label": "x"})            # no token
    assert code == 403
    code, r = _post(port, "/api/workflow/add", {"label": "x"}, {"X-Carto-Token": tok})
    assert code == 200 and r["ok"]


def test_locked_delete_returns_423(server):
    port, tok = server
    code, r = _post(port, "/api/workflow/delete", {"id": "redact"}, {"X-Carto-Token": tok})
    assert code == 423 and r["locked"] and r["disclaimer"]


def test_disallowed_origin_not_echoed(server):
    port, _ = server
    _, _, h = _get(port, "/api/workflow", {"Origin": "https://evil.example"})
    assert h.get("Access-Control-Allow-Origin") in (None,)                # never echoed for bad origin
