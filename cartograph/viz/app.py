"""Local server for the graph visualizer + the Studio editor. Stdlib only (http.server).

Hardened for V1 (see cartograph/security.py): binds to loopback, a strict CORS allowlist (never `*`),
a DNS-rebinding Host check, and an API token required for every state-changing endpoint. Read endpoints
expose only your own local data; mutations only touch the workflow OVERLAY (never the graph DB).

    carto viz       # the graph visualizer
    carto studio    # the editable workflow Studio
    carto serve     # headless API (for the web-GenAI userscript)
"""
from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import db_path, load_config
from ..security import api_token, cors_origin, host_is_local, token_ok
from ..storage import Store

_HTML = Path(__file__).with_name("index.html")
_STUDIO = Path(__file__).with_name("studio.html")
_STATIC = {"/mascot.png": "image/png", "/favicon.png": "image/png",
           "/moley.svg": "image/svg+xml", "/favicon.svg": "image/svg+xml"}


def _q(u, key: str, default: str = "") -> str:
    return (parse_qs(u.query).get(key) or [default])[0]


def entity_hubs(store: Store, project_ids: set[int], limit: int = 60) -> dict:
    """Entities shared by several projects (papers, repos, models, packages), as extra Atlas nodes."""
    if not store.has_table("mentions") or not project_ids:
        return {"nodes": [], "edges": []}
    with store.cursor() as c:
        rows = c.execute(
            "SELECT e.id, e.kind, e.name, f.project_id pid, sum(m.count) n FROM mentions m "
            "JOIN entities e ON e.id = m.entity_id JOIN files f ON f.id = m.file_id "
            "GROUP BY e.id, f.project_id").fetchall()
    from ..entities import _TOOLCHAIN
    by_ent: dict[int, dict] = {}
    for r in rows:
        if r["pid"] not in project_ids or (r["kind"] == "package" and r["name"] in _TOOLCHAIN):
            continue
        d = by_ent.setdefault(r["id"], {"kind": r["kind"], "name": r["name"], "projects": {}})
        d["projects"][r["pid"]] = r["n"]
    shared = [(k, v) for k, v in by_ent.items() if 2 <= len(v["projects"]) <= max(3, len(project_ids) // 3)]
    rarity = {"paper": 3.0, "dataset": 2.5, "model": 2.5, "repo": 1.5, "package": 1.0}
    shared.sort(key=lambda kv: -(rarity.get(kv[1]["kind"], 1.0) * len(kv[1]["projects"])))
    nodes, edges = [], []
    for eid, d in shared[:limit]:
        nid = f"e{eid}"
        nodes.append({"id": nid, "label": d["name"], "kind": "entity", "entity_kind": d["kind"],
                      "size": 4 + 2 * len(d["projects"])})
        edges += [{"source": f"p{pid}", "target": nid, "relation": "MENTIONS", "weight": n}
                  for pid, n in d["projects"].items()]
    return {"nodes": nodes, "edges": edges}


def project_detail(store: Store, node_id: str) -> dict:
    from .. import entities as E
    from .. import memory as M
    pid = int(node_id.lstrip("p"))
    with store.cursor() as c:
        row = c.execute("SELECT id, name, path, field FROM projects WHERE id = ?", (pid,)).fetchone()
        if not row:
            return {"error": "no such project"}
        nf = c.execute("SELECT count(*) FROM files WHERE project_id = ?", (pid,)).fetchone()[0]
    out = {"id": node_id, "name": row["name"], "path": row["path"], "field": row["field"] or "", "files": nf,
           "related": E.related_projects(store, pid)[:12], "entities": E.project_entities(store, pid, limit=40)}
    out["memories"] = M.recall(store, cwd=row["path"], history=True, limit=30)["memories"] \
        if store.has_table("memories") and row["path"] else []
    return out


_ATLAS: dict = {}
_ATLAS_LOCK = threading.Lock()


def _db_version() -> tuple:
    p = db_path()
    return tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in (p, p.with_name(p.name + "-wal")))


def atlas_json() -> str:
    """The Atlas payload, rebuilt only when the graph file changes (counting 6M+ chunks takes seconds)."""
    with _ATLAS_LOCK:
        ver = _db_version()
        if _ATLAS.get("version") != ver:
            from .. import entities as E
            from .. import memory as M
            store = Store(db_path(), read_only=True)
            data = store.graph_sample(max_nodes=600)
            hubs = entity_hubs(store, {int(n["id"][1:]) for n in data["nodes"]})
            data["nodes"] += hubs["nodes"]
            data["edges"] += hubs["edges"]
            data["stats"] = {**store.stats(), **E.counts(store), **M.stats(store)}
            _ATLAS.update(version=ver, body=json.dumps(data))
        return _ATLAS["body"]


def timeline(store: Store, limit: int = 200) -> dict:
    from .. import memory as M
    if not store.has_table("memories"):
        return {"memories": []}
    return {"memories": M.recall(store, history=True, limit=limit)["memories"]}


def _chat_command(text: str) -> dict:
    """Tiny built-in command grammar so the Studio chat can manipulate the workflow with NO LLM attached.
    Connecting a real local/cloud LLM is opt-in and gated by a privacy disclaimer in the UI."""
    from .. import workflow
    t = (text or "").strip()
    low = t.lower()
    try:
        if low.startswith("add node "):
            return {"reply": "Added node.", "result": workflow.add_node(t[9:].strip())}
        if low.startswith("import model "):
            rest = t[13:].strip()
            name, _, src = rest.partition(" from ")
            return {"reply": f"Imported '{name.strip()}' as a hovering node — drag it between two nodes "
                    "to wire it in.", "result": workflow.import_model(name.strip(), src.strip() or "(local)")}
        if low.startswith("delete node ") or low.startswith("remove node "):
            r = workflow.delete_node(t.split(" ", 2)[2].strip())
            return {"reply": r.get("disclaimer") or "Done.", "result": r}
        if low.startswith("connect ") and " to " in low:
            a, _, b = t[8:].partition(" to ")
            return {"reply": "Connected.", "result": workflow.connect(a.strip(), b.strip())}
        if low.startswith("details ") or low.startswith("explain "):
            return {"reply": "Here are the node details.", "result": workflow.node_details(t.split(" ", 1)[1].strip())}
        if low in ("stats", "show stats"):
            from ..insights import simple_stats
            return {"reply": "Current stats.", "result": simple_stats()}
    except Exception as e:  # noqa: BLE001
        return {"reply": f"Couldn't run that: {e}", "result": {"ok": False}}
    return {"reply": "No model is connected, so I understand only built-in commands: 'add node X', "
            "'import model NAME from SRC', 'delete node ID', 'connect A to B', 'details ID', 'stats'. "
            "Connect a local/cloud LLM (with the privacy disclaimer) for free-form graph editing.",
            "result": {"ok": True, "builtin": True}}


def _make_handler():
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            origin = cors_origin(self.headers.get("Origin"))
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if origin:                                   # echo ONLY allowlisted origins, never `*`
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Carto-Token")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

        def _guard(self) -> bool:
            if not host_is_local(self.headers.get("Host")):     # DNS-rebinding defense
                self._send(403, json.dumps({"error": "non-local host refused"}))
                return False
            return True

        def do_OPTIONS(self):
            if self._guard():
                self._send(204, b"")

        def do_GET(self):
            if not self._guard():
                return
            u = urlparse(self.path)
            ro = lambda: Store(db_path(), read_only=True)  # noqa: E731
            if u.path in ("/", "/index.html", "/atlas"):
                return self._send(200, _HTML.read_bytes(), "text/html; charset=utf-8")
            if u.path in _STATIC:
                f = Path(__file__).with_name(u.path.lstrip("/"))
                return self._send(200, f.read_bytes(), _STATIC[u.path]) if f.exists() else self._send(404, b"")
            if u.path == "/api/atlas":
                return self._send(200, atlas_json())
            if u.path == "/api/context":
                try:
                    from ..retrieve import context
                    out = context(_q(u, "q"), ro(), load_config(), mode=_q(u, "mode", "mix"),
                                  max_tokens=int(_q(u, "max_tokens", "2500")), top_k=10)
                    return self._send(200, json.dumps(out, default=str))
                except Exception as e:  # noqa: BLE001
                    return self._send(200, json.dumps({"error": str(e)}))
            if u.path == "/api/project":
                try:
                    return self._send(200, json.dumps(project_detail(ro(), _q(u, "id")), default=str))
                except (ValueError, TypeError) as e:
                    return self._send(400, json.dumps({"error": str(e)}))
            if u.path == "/api/timeline":
                return self._send(200, json.dumps(timeline(ro(), int(_q(u, "limit", "200"))), default=str))
            if u.path == "/studio":
                html = _STUDIO.read_text(encoding="utf-8").replace("__CARTO_TOKEN__", api_token())
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/api/graph":
                store = ro()
                data = store.graph_sample(max_nodes=400)
                data["stats"] = store.stats()
                return self._send(200, json.dumps(data))
            if u.path == "/api/workflow":
                from .. import workflow
                return self._send(200, json.dumps(workflow.get_workflow()))
            if u.path == "/api/graphs":
                from .. import workflow
                return self._send(200, json.dumps({"graphs": workflow.list_graphs()}))
            if u.path == "/api/node":
                from .. import workflow
                nid = (parse_qs(u.query).get("id") or [""])[0]
                return self._send(200, json.dumps(workflow.node_details(nid)))
            if u.path == "/api/stats":
                from ..insights import advanced_stats, simple_stats
                qs = parse_qs(u.query)
                adv = (qs.get("advanced") or ["0"])[0] in ("1", "true", "yes")
                deep = (qs.get("deep") or ["0"])[0] in ("1", "true", "yes")
                return self._send(200, json.dumps(advanced_stats(deep=deep) if adv else simple_stats()))
            if u.path == "/api/search":
                q = (parse_qs(u.query).get("q") or [""])[0]
                try:
                    from ..retrieve import retrieve
                    res = retrieve(q, ro(), load_config(), top_k=12)
                    return self._send(200, json.dumps({"method": res.method, "projects": res.projects,
                                                       "chunks": res.chunks[:8]}))
                except Exception as e:
                    return self._send(200, json.dumps({"error": str(e), "projects": [], "chunks": []}))
            if u.path == "/api/personalize":
                q = (parse_qs(u.query).get("prompt") or parse_qs(u.query).get("q") or [""])[0]
                try:
                    from ..persona import build_brief
                    from ..persona.profile import load_persona
                    store = ro()
                    return self._send(200, json.dumps(build_brief(q, store, load_config(), load_persona(store))))
                except Exception as e:
                    return self._send(200, json.dumps({"error": str(e), "output_guidance": []}))
            return self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if not self._guard():
                return
            u = urlparse(self.path)
            try:
                ln = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(ln) or b"{}") if ln else {}
            except Exception:
                return self._send(400, json.dumps({"error": "bad json"}))
            token = self.headers.get("X-Carto-Token") or body.get("token")
            if not token_ok(token):                      # every mutation requires the workspace token
                return self._send(403, json.dumps({"error": "invalid or missing token"}))
            from .. import workflow
            p = u.path
            try:
                if p == "/api/workflow/add":
                    r = workflow.add_node(body.get("label", ""), body.get("kind", "custom"), body.get("desc", ""))
                elif p == "/api/workflow/import-model":
                    r = workflow.import_model(body.get("name", ""), body.get("source", ""),
                                              body.get("kind", "model"), body.get("desc", ""))
                elif p == "/api/workflow/delete":
                    r = workflow.delete_node(body.get("id", ""))
                elif p == "/api/workflow/enable":
                    r = workflow.enable_node(body.get("id", ""))
                elif p == "/api/workflow/edge":
                    r = workflow.connect(body.get("from", ""), body.get("to", ""))
                elif p == "/api/workflow/unedge":
                    r = workflow.disconnect(body.get("from", ""), body.get("to", ""))
                elif p == "/api/workflow/position":
                    r = workflow.set_position(body.get("id", ""), body.get("x", 0), body.get("y", 0))
                elif p == "/api/chat":
                    r = _chat_command(body.get("text", ""))
                else:
                    return self._send(404, json.dumps({"error": "not found"}))
            except Exception as e:  # noqa: BLE001
                return self._send(200, json.dumps({"ok": False, "error": str(e)}))
            code = 423 if (isinstance(r, dict) and r.get("locked") and not r.get("ok")) else 200
            return self._send(code, json.dumps(r))
    return Handler


def launch(host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True, path: str = "/") -> None:
    if not db_path().exists():
        print("No graph yet. Run `carto init` then `carto ingest <folder>` first.")
        return
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    for p in range(port, port + 20):
        try:
            httpd = Server((host, p), _make_handler())
            break
        except OSError:
            continue
    else:
        print(f"Ports {port}-{port + 19} are all in use; pass --port.")
        return
    if p != port:
        print(f"port {port} is busy, using {p}")
    url = f"http://{host}:{p}{path}"
    threading.Thread(target=atlas_json, daemon=True).start()
    print(f"Cartograph running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        httpd.shutdown()
