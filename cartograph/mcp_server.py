"""Cartograph MCP server — plug your graph into any MCP client (Claude Code, Cursor, Codex, Windsurf, ...).
Stdio JSON-RPC; speaks the standard `initialize` / `tools/list` / `tools/call` methods.

Retrieval tools only read your graph. Tools that write local state (persona feedback, memories) say so
in their MCP `annotations` (readOnlyHint=false); `carto mcp-server --read-only` hides them entirely.

Register (Claude Code / Cursor `mcp.json`):
    { "mcpServers": { "cartograph": { "command": "carto", "args": ["mcp-server"], "type": "stdio" } } }
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

from .config import db_path, load_config
from .storage import Store

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

_STR = {"type": "string"}
_INT = {"type": "integer"}
_STRS = {"type": "array", "items": {"type": "string"}}


def _obj(props: dict, required: tuple[str, ...] = ()) -> dict:
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = list(required)
    return schema


def _d(schema: dict, description: str, **extra) -> dict:
    return {**schema, "description": description, **extra}


TOOLS = [
    {"name": "retrieve_context",
     "description": "BEST retrieval for injecting context: hybrid (semantic + keyword) over the user's whole "
                    "graph — code, notes, PDFs, Office docs. Every snippet has a `cite` (file:L40-88, paper.pdf "
                    "p.12). Set mode=mix with max_tokens for one budgeted bundle of relevant memories, entity "
                    "facts and snippets.",
     "schema": _obj({"query": _d(_STR, "What to find."),
                     "top_k": _d(_INT, "How many snippets (default 8).", minimum=1, maximum=50),
                     "mode": _d(_STR, "chunks (default): snippets. mix: memories + entity facts + snippets. "
                                      "graph: entity facts. memory: agent memories. projects: ranked projects.",
                                enum=["chunks", "mix", "graph", "memory", "projects"]),
                     "max_tokens": _d(_INT, "Hard token budget for the whole result (default 2000 when mode is set).",
                                      minimum=200, maximum=50000),
                     "cwd": _d(_STR, "Working directory, scopes memories to its project.")}, ("query",))},
    {"name": "relevant_projects",
     "description": "Which of your projects/repos are most relevant to a query (hybrid ranked).",
     "schema": _obj({"query": _d(_STR, "What the task is about."),
                     "top_k": _d(_INT, "How many projects (default 8).", minimum=1, maximum=50)}, ("query",))},
    {"name": "elevate_task",
     "description": "One-shot top-of-field briefing for a build task: elite bar, reference repos, the "
                    "frontier playbook (process), adjoining-field moves, and your most relevant existing repos.",
     "schema": _obj({"task": _d(_STR, "The build task."),
                     "project_path": _d(_STR, "Optional path of the project being built.")}, ("task",))},
    {"name": "frontier_status",
     "description": "Coverage of each field's top-tier reference repos in your graph + the prioritized "
                    "acquisition backlog.",
     "schema": _obj({"top": _d(_INT, "How many fields to report (default 8).", minimum=1, maximum=50)})},
    {"name": "personalize",
     "description": "THE personalization hook — call FIRST on any user prompt. Returns a steering brief "
                    "(the user's persona/field focus, output guidance, the compass constellation of "
                    "ambitions, ventures, inquiries, sparks, and sorties, and persona-ranked snippets) "
                    "to prepend so your answer fits this specific user. They hold many directions at "
                    "once; do not collapse them onto one project. Adapts as feedback accrues.",
     "schema": _obj({"prompt": _d(_STR, "The user's request."),
                     "top_k": _d(_INT, "How many snippets (default 6).", minimum=1, maximum=50),
                     "max_chars": _d(_INT, "Context budget in characters (0 = unlimited).", minimum=0)},
                    ("prompt",))},
    {"name": "constellation",
     "description": "The user's compass: slow ambitions, warm ventures and inquiries, new sparks, "
                    "temporary sorties and their residue, threads they returned to, and resting work "
                    "that is memory rather than a to-do. Pass a prompt to mark which threads it touches, "
                    "including quiet ones. Built by `carto compass --sync` from AutoLog.",
     "schema": _obj({"prompt": _d(_STR, "Optional prompt to mark which threads it touches.")})},
    {"name": "record_use", "writes": True,
     "description": "Close the learning loop: after answering, report which projects actually helped "
                    "(and which didn't) so the persona adapts to what the user responds to. Call when you "
                    "used Cartograph context in a useful answer. Writes local persona feedback.",
     "schema": _obj({"query": _d(_STR, "The query the feedback is about."),
                     "helped": _d(_STRS, "Project names that helped."),
                     "not_helpful": _d(_STRS, "Project names that did not help.")}, ("helped",))},
    {"name": "frontier_hints",
     "description": "Reasoning hints for the task at hand: the textbook method for its technical area "
                    "(CUDA, time series, MD, signals, sparse, HPC, quant, ML), one worth-trying tangent with "
                    "when-to-try, frontier papers/repos the scout pulled for the user's active topics, and "
                    "related projects from the user's graph. Use when planning an implementation.",
     "schema": _obj({"prompt": _d(_STR, "The task or question.")}, ("prompt",))},
    {"name": "entities",
     "description": "Facts read deterministically from the user's files, each with evidence (a citable "
                    "chunk), extractor, confidence and first/last seen: dependencies, imports, cited papers "
                    "(DOI/arXiv), GitHub repos, Hugging Face models/datasets. Give `find` to see which "
                    "projects use/cite something (e.g. 'pydantic', '1706.03762'); give `project` to list "
                    "what one project depends on and which projects it shares entities with.",
     "schema": _obj({"find": _d(_STR, "Entity name or fragment to look up across projects."),
                     "project": _d(_STR, "Project name to describe."),
                     "kind": _d(_STR, "Restrict to one kind.", enum=["package", "paper", "repo", "model", "dataset"]),
                     "limit": _d(_INT, "Max facts (default 20).", minimum=1, maximum=200)})},
    {"name": "brief",
     "description": "Session-start memory digest for the project being worked on: its active decisions, "
                    "rejected approaches (don't retry them), preferences, open todos, recent sessions, and "
                    "memories whose anchored files changed. Call at the start of a task. Bounded by max_chars.",
     "schema": _obj({"cwd": _d(_STR, "The working directory (picks the project)."),
                     "project": _d(_STR, "Project name, if known."),
                     "max_chars": _d(_INT, "Character budget (default 2000).", minimum=200, maximum=20000)})},
    {"name": "recall",
     "description": "Search the user's shared agent memory (decisions, rejected approaches, preferences, facts, "
                    "todos, session logs). Active memories only unless history=true; each result says if it "
                    "may be stale because a file it depends on changed.",
     "schema": _obj({"query": _d(_STR, "Keywords (empty = newest)."),
                     "project": _d(_STR, "Project name."), "cwd": _d(_STR, "Working directory (picks the project)."),
                     "kinds": _d(_STRS, "Restrict to kinds: decision, rejected, preference, fact, todo, note, session."),
                     "history": _d({"type": "boolean"}, "Include superseded memories."),
                     "limit": _d(_INT, "Max results (default 10).", minimum=1, maximum=100),
                     "max_chars": _d(_INT, "Cap on total text returned (0 = none).", minimum=0)})},
    {"name": "remember", "writes": True,
     "description": "Save something every future session (in any client) should know: a decision and why, an "
                    "approach that was tried and rejected, a user preference, a fact, a todo. Anchor it to the "
                    "files it depends on so it's flagged stale when they change. Returns similar existing "
                    "memories; supersede them if this one replaces them.",
     "schema": _obj({"text": _d(_STR, "The memory, self-contained (include the why)."),
                     "kind": _d(_STR, "Memory kind.", enum=["decision", "rejected", "preference", "fact", "todo", "note"]),
                     "project": _d(_STR, "Project name."), "cwd": _d(_STR, "Working directory (picks the project)."),
                     "anchors": _d(_STRS, "File paths this memory depends on."),
                     "supersedes": _d({"type": "array", "items": {"type": "integer"}}, "Memory ids this replaces."),
                     "source": _d(_STR, "Client name, e.g. claude-code, cursor.")}, ("text",))},
    {"name": "supersede", "writes": True,
     "description": "Mark a memory as no longer true (kept as history with its validity interval), optionally "
                    "pointing at the memory that replaces it.",
     "schema": _obj({"id": _d(_INT, "Memory to close."), "by": _d(_INT, "Replacement memory id."),
                     "reason": _d(_STR, "Why it no longer holds.")}, ("id",))},
    {"name": "graph_stats",
     "description": "Counts in your graph (projects/files/chunks/edges/entities) and the semantic index state.",
     "schema": _obj({})},
]
_BY_NAME = {t["name"]: t for t in TOOLS}
_VALID = set(_BY_NAME)


def _annotations(t: dict) -> dict:
    writes = bool(t.get("writes"))
    return {"title": t["name"].replace("_", " "), "readOnlyHint": not writes, "destructiveHint": False,
            "idempotentHint": not writes, "openWorldHint": False}


def read_only() -> bool:
    return os.environ.get("CARTOGRAPH_MCP_READ_ONLY", "").strip().lower() in ("1", "true", "yes")


def tool_list() -> list[dict]:
    ro = read_only()
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["schema"],
             "annotations": _annotations(t)} for t in TOOLS if not (ro and t.get("writes"))]


def _resp(req_id: Any, result: Any = None, error: str | None = None, code: int = -32000) -> str:
    body: dict = {"jsonrpc": "2.0", "id": req_id}
    if error:
        body["error"] = {"code": code, "message": error}
    else:
        body["result"] = result
    return json.dumps(body)


def _names(v) -> list[str]:
    """Accept a list, a JSON-array string or a comma-separated string (lenient clients send all three)."""
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                v = s.strip("[]").split(",")
        else:
            v = s.split(",")
    return [str(x).strip().strip('"') for x in v if str(x).strip()]


def _negotiate(requested: Any) -> str:
    return requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]


def _handle(req: dict) -> str | None:
    method = req.get("method")
    params = req.get("params") or {}
    rid = req.get("id")

    if method == "initialize":
        from . import __version__
        return _resp(rid, {"protocolVersion": _negotiate(params.get("protocolVersion")),
                           "capabilities": {"tools": {"listChanged": False}},
                           "serverInfo": {"name": "cartograph", "version": __version__}})
    if isinstance(method, str) and method.startswith("notifications/"):
        return None                                       # JSON-RPC notifications get no response
    if method == "ping":
        return _resp(rid, {})
    if method == "tools/list":
        return _resp(rid, {"tools": tool_list()})
    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        if name not in _VALID:
            return _resp(rid, error=f"unknown tool: {name}", code=-32602)
        if read_only() and _BY_NAME[name].get("writes"):
            return _resp(rid, error=f"{name} writes local state; this server runs read-only", code=-32602)
        obj = json.loads(_dispatch(rid, name, args))
        if "error" in obj:                                # tool failures are results, not protocol errors
            return _resp(rid, {"content": [{"type": "text", "text": obj["error"].get("message", "tool error")}],
                               "isError": True})
        return _resp(rid, {"content": [{"type": "text", "text": json.dumps(obj.get("result"))}]})
    if method in _VALID:  # allow bare method calls too
        return _dispatch(rid, method, params)
    return _resp(rid, error=f"unknown method: {method}", code=-32601)


def _dispatch(rid: Any, name: str, args: dict) -> str:
    """Run a tool. A long-lived server can outlive a `git pull`/reinstall of Cartograph; if an import
    then fails on stale finder caches, invalidate them and retry once instead of failing forever."""
    try:
        return _run(rid, name, args)
    except ImportError:
        importlib.invalidate_caches()
        for mod in [m for m in sys.modules if m.startswith("cartograph.") and sys.modules[m] is None]:
            del sys.modules[mod]
        try:
            return _run(rid, name, args)
        except Exception as e:  # noqa: BLE001 - never hang the client
            return _resp(rid, error=f"{name} failed: {e}")
    except Exception as e:  # noqa: BLE001 - never hang the client
        return _resp(rid, error=f"{name} failed: {e}")


def _run(rid: Any, name: str, args: dict) -> str:
    if not db_path().exists():
        return _resp(rid, error="No Cartograph graph yet. Run `carto init` and `carto ingest` first.")
    cfg = load_config()
    store = Store(db_path(), read_only=True)
    if name == "graph_stats":
        from .embed import index_status
        st = index_status()
        from .entities import counts
        from .memory import stats as mstats
        return _resp(rid, {**store.stats(), **counts(store), **mstats(store),
                           "index": {k: st.get(k) for k in ("state", "done", "total", "running")}})
    if name == "entities":
        from . import entities as E
        limit, kind = int(args.get("limit", 20)), args.get("kind") or None
        if args.get("find"):
            return _resp(rid, {"facts": E.find(store, args["find"], kind=kind, limit=limit)})
        if args.get("project"):
            pid = store.project_id_by_name(args["project"])
            if pid is None:
                return _resp(rid, error=f"no project named {args['project']!r}")
            return _resp(rid, {"facts": E.project_entities(store, pid, kind=kind, limit=limit),
                               "related": E.related_projects(store, pid)})
        return _resp(rid, error="give `find` or `project`")
    if name in ("retrieve_context", "relevant_projects"):
        from .retrieve import retrieve
        query = args.get("query", "")
        if not query:
            return _resp(rid, error="query is required")
        if name == "retrieve_context" and (args.get("mode") or args.get("max_tokens")):
            from .retrieve import context
            return _resp(rid, context(query, store, cfg, mode=args.get("mode") or "chunks",
                                      max_tokens=int(args.get("max_tokens") or 2000),
                                      top_k=int(args.get("top_k", 8)), cwd=args.get("cwd")))
        res = retrieve(query, store, cfg, top_k=int(args.get("top_k", 8)))
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
    if name == "frontier_hints":
        from .elite.methods import match_methods
        from .scout import frontier_cards
        prompt = args.get("prompt", "")
        out = {"methods": match_methods(prompt), "frontier": frontier_cards(store, prompt, limit=4)}
        try:
            from .persona import personalized_retrieve
            from .persona.profile import load_persona
            out["related"] = personalized_retrieve(prompt, store, cfg, load_persona(store), top_k=6).get("related", [])
        except Exception:
            out["related"] = []
        return _resp(rid, out)
    if name == "constellation":
        from .compass import compass_for_prompt
        return _resp(rid, compass_for_prompt(store, args.get("prompt") or ""))
    if name in ("brief", "recall", "remember", "supersede"):
        from . import memory as M
        if name == "brief":
            return _resp(rid, M.brief(store, project=args.get("project"), cwd=args.get("cwd"),
                                      max_chars=int(args.get("max_chars", 2000))))
        if name == "recall":
            return _resp(rid, M.recall(store, args.get("query") or "", project=args.get("project"),
                                       cwd=args.get("cwd"), kinds=_names(args.get("kinds")) or None,
                                       history=bool(args.get("history")), limit=int(args.get("limit", 10)),
                                       max_chars=int(args.get("max_chars", 0))))
        rw = Store(db_path())
        if name == "remember":
            return _resp(rid, M.remember(rw, args.get("text", ""), args.get("kind") or "note",
                                         project=args.get("project"), cwd=args.get("cwd"),
                                         anchors=_names(args.get("anchors")), source=args.get("source") or "mcp",
                                         supersedes=[int(x) for x in _names(args.get("supersedes"))]))
        return _resp(rid, M.supersede(rw, int(args["id"]), args.get("by"), args.get("reason") or ""))
    if name == "record_use":
        from .persona import record_feedback
        from .persona.profile import load_persona
        p = record_feedback(load_persona(store), store, cfg, query=args.get("query", ""),
                            liked_projects=_names(args.get("helped")),
                            disliked_projects=_names(args.get("not_helpful")))
        return _resp(rid, {"ok": True, "n_signals": p.n_signals, "persona": p.summary()})
    return _resp(rid, error=f"unhandled: {name}")


def _warm() -> None:
    """Load the embedding model and vector index before the first tool call asks for them (8-10 s cold)."""
    try:
        from .retrieve import _semantic_chunk_ids
        _semantic_chunk_ids(load_config(), "warm up", 1)
    except Exception:
        pass


def serve(read_only_mode: bool = False) -> None:
    if read_only_mode:
        os.environ["CARTOGRAPH_MCP_READ_ONLY"] = "1"
    if os.environ.get("CARTOGRAPH_MCP_WARM", "1") != "0":
        import threading
        threading.Thread(target=_warm, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(_resp(None, error="parse error", code=-32700) + "\n")
            sys.stdout.flush()
            continue
        is_notification = "id" not in req                 # JSON-RPC notifications get no response
        out = _handle(req)
        if out is not None and not is_notification:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve("--read-only" in sys.argv)
