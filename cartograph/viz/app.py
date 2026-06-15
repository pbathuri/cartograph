"""Desktop graph visualizer — zero extra dependencies (Python stdlib http.server). Serves an
interactive force-directed graph of your projects + a live search box, and opens your browser.

    carto viz                  # launches at http://127.0.0.1:8787 and opens the browser

Built for non-technical users: one command (or one double-click via scripts/launch_viz), a visual
map you can pan/zoom/click, and a search that highlights relevant projects."""
from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import db_path, load_config
from ..storage import Store

_HTML = Path(__file__).with_name("index.html")


def _make_handler():
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, _HTML.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/graph":
                store = Store(db_path(), read_only=True)
                data = store.graph_sample(max_nodes=400)
                data["stats"] = store.stats()
                return self._send(200, json.dumps(data))
            if u.path == "/api/search":
                q = (parse_qs(u.query).get("q") or [""])[0]
                store = Store(db_path(), read_only=True)
                try:
                    from ..retrieve import retrieve
                    res = retrieve(q, store, load_config(), top_k=12)
                    return self._send(200, json.dumps({"method": res.method, "projects": res.projects,
                                                       "chunks": res.chunks[:8]}))
                except Exception as e:
                    return self._send(200, json.dumps({"error": str(e), "projects": [], "chunks": []}))
            return self._send(404, json.dumps({"error": "not found"}))
    return Handler


def launch(host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True) -> None:
    if not db_path().exists():
        print("No graph yet. Run `carto init` then `carto ingest <folder>` first.")
        return
    httpd = socketserver.ThreadingTCPServer((host, port), _make_handler())
    url = f"http://{host}:{port}"
    print(f"Cartograph viz running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        httpd.shutdown()
