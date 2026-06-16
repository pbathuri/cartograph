"""Cartograph MCP server — plug your graph into any MCP client (Claude Code, Cursor, etc.).
Stdio JSON-RPC; speaks the standard `initialize` / `tools/list` / `tools/call` methods. Read-only.

Register (Claude Code / Cursor `mcp.json`):
    { "mcpServers": { "cartograph": { "command": "carto", "args": ["mcp-server"], "type": "stdio" } } }
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .config import db_path, load_config
from .storage import Store

TOOLS = [
    {"name": "retrieve_context",
     "description": "BEST retrieval for injecting context: hybrid (semantic + keyword) over your whole "
                    "graph. Returns the relevant code/doc snippets (text+file+project). Use this to pull "
                    "context into a prompt.",
     "params": {"query": "string", "top_k": "int? (default 8)"}},
    {"name": "relevant_projects",
     "description": "Which of your projects/repos are most relevant to a query (hybrid ranked).",
     "params": {"query": "string", "top_k": "int? (default 8)"}},
    {"name": "elevate_task",
     "description": "One-shot top-of-field briefing for a build task: elite bar, reference repos, the "
                    "frontier playbook (process), adjoining-field moves, and your most relevant existing repos.",
     "params": {"task": "string", "project_path": "string?"}},
    {"name": "frontier_status",
     "description": "Coverage of each field's top-tier reference repos in your graph + the prioritized "
                    "acquisition backlog.",
     "params": {"top": "int? (default 8)"}},
    {"name": "personalize",
     "description": "THE personalization hook — call FIRST on any user prompt. Returns a steering brief "
                    "(the user's persona/field focus, output guidance, and persona-ranked relevant "
                    "snippets) to prepend so your answer fits this specific user and their work. Adapts "
                    "as feedback accrues.",
     "params": {"prompt": "string", "top_k": "int? (default 6)", "max_chars": "int? context budget (0=unlimited)"}},
    {"name": "record_use",
     "description": "Close the learning loop: after answering, report which projects actually helped "
                    "(and which didn't) so the persona adapts to what the user responds to. Call when you "
                    "used Cartograph context in a useful answer.",
     "params": {"query": "string?", "helped": "string[] project names",
                "not_helpful": "string[]? project names"}},
    {"name": "graph_stats",
     "description": "Counts in your graph (projects/files/chunks/edges).", "params": {}},
]
_VALID = {t["name"] for t in TOOLS}


def _resp(req_id: Any, result: Any = None, error: str | None = None) -> str:
    body: dict = {"jsonrpc": "2.0", "id": req_id}
    if error:
        body["error"] = {"code": -32000, "message": error}
    else:
        body["result"] = result
    return json.dumps(body)


def _params_to_schema(params: dict | None) -> dict:
    """Shorthand `params` -> JSON-Schema `inputSchema` (MCP-spec; strict clients like Cursor require it)."""
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in (params or {}).items():
        s = str(spec).lower()
        t = ("integer" if "int" in s else "number" if ("float" in s or "number" in s)
             else "boolean" if "bool" in s else "string")
        props[name] = {"type": t, "description": str(spec)}
        if "?" not in s and "default" not in s:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _handle(req: dict) -> str | None:
    method = req.get("method")
    params = req.get("params") or {}
    rid = req.get("id")

    if method == "initialize":
        from . import __version__
        return _resp(rid, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "cartograph", "version": __version__}})
    if isinstance(method, str) and method.startswith("notifications/"):
        return None                                       # JSON-RPC notifications get no response
    if method == "tools/list":
        tools = [{"name": t["name"], "description": t["description"],
                  "inputSchema": _params_to_schema(t.get("params"))} for t in TOOLS]
        return _resp(rid, {"tools": tools})
    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        if name not in _VALID:
            return _resp(rid, error=f"unknown tool: {name}")
        inner = _dispatch(rid, name, args)
        obj = json.loads(inner) if inner else {}
        if "error" in obj:
            return _resp(rid, error=obj["error"].get("message", "tool error"))
        return _resp(rid, {"content": [{"type": "text", "text": json.dumps(obj.get("result"))}]})
    if method in _VALID:  # allow bare method calls too
        return _dispatch(rid, method, params)
    return _resp(rid, error=f"unknown method: {method}")


def _dispatch(rid: Any, name: str, args: dict) -> str:
    cfg = load_config()
    store = Store(db_path(), read_only=True)
    try:
        if name == "graph_stats":
            return _resp(rid, store.stats())
        if name in ("retrieve_context", "relevant_projects"):
            from .retrieve import retrieve
            res = retrieve(args.get("query", ""), store, cfg, top_k=int(args.get("top_k", 8)))
            if name == "relevant_projects":
                return _resp(rid, {"method": res.method, "projects": res.projects})
            return _resp(rid, {"method": res.method, "chunks": res.chunks})
        if name == "elevate_task":
            from .elite import elevate
            task = args.get("task", "")
            if not task:
                return _resp(rid, error="task is required")
            return _resp(rid, elevate(task, store, cfg, project=args.get("project_path")))
        if name == "frontier_status":
            from .elite import frontier_report
            return _resp(rid, frontier_report(store, top=int(args.get("top", 8))))
        if name == "personalize":
            from .persona import build_brief
            from .persona.profile import load_persona
            prompt = args.get("prompt", "")
            if not prompt:
                return _resp(rid, error="prompt is required")
            return _resp(rid, build_brief(prompt, store, cfg, load_persona(store),
                                          top_k=int(args.get("top_k", 6)),
                                          max_chars=int(args.get("max_chars", 0))))
        if name == "record_use":
            from .persona import record_feedback
            from .persona.profile import load_persona
            p = record_feedback(load_persona(store), store, cfg, query=args.get("query", ""),
                                liked_projects=list(args.get("helped", []) or []),
                                disliked_projects=list(args.get("not_helpful", []) or []))
            return _resp(rid, {"ok": True, "n_signals": p.n_signals, "persona": p.summary()})
    except Exception as e:  # noqa: BLE001 - never hang the client
        return _resp(rid, error=f"{name} failed: {e}")
    return _resp(rid, error=f"unhandled: {name}")


def serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        is_notification = "id" not in req                 # JSON-RPC notifications get no response
        out = _handle(req)
        if out is not None and not is_notification:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve()
