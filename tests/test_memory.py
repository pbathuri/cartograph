"""Temporal agent memory + automatic session capture through client hooks."""
import json

import pytest

from cartograph import hooks as H
from cartograph import memory as M
from cartograph.config import Config, db_path
from cartograph.ingest import ingest_path
from cartograph.mcp_server import _handle
from cartograph.storage import Store


@pytest.fixture
def proj(tmp_path):
    d = tmp_path / "ocular"
    d.mkdir()
    (d / ".git").mkdir()
    (d / "model.py").write_text("k_el = 0.2\n")
    store = Store(db_path())
    ingest_path(d, store, Config())
    return d, store


def test_remember_recall_and_project_from_cwd(proj):
    d, store = proj
    m = M.remember(store, "Use a two-compartment model for timolol; one-compartment underfit the AH data",
                   "decision", cwd=str(d / "sub"))
    assert m["project"] == "ocular" and m["possible_conflicts"] == []
    hits = M.recall(store, "timolol compartment", cwd=str(d))["memories"]
    assert [h["id"] for h in hits] == [m["id"]]
    again = M.remember(store, m["text"], "decision", cwd=str(d))
    assert again["duplicate"] and again["id"] == m["id"]


def test_supersede_keeps_history_and_flags_conflicts(proj):
    d, store = proj
    old = M.remember(store, "Store vectors in a numpy memmap index", "decision", cwd=str(d))
    new = M.remember(store, "Store vectors in a HNSW index instead of the numpy memmap", "decision", cwd=str(d))
    assert [c["id"] for c in new["possible_conflicts"]] == [old["id"]]
    M.supersede(store, old["id"], new["id"], reason="memmap too slow at 6M rows")
    active = M.recall(store, "vectors index", cwd=str(d))["memories"]
    assert [a["id"] for a in active] == [new["id"]]
    hist = {h["id"]: h for h in M.recall(store, "vectors index", cwd=str(d), history=True)["memories"]}
    assert hist[old["id"]]["superseded_by"] == new["id"] and hist[old["id"]]["valid_to"]
    assert hist[old["id"]]["reason"] == "memmap too slow at 6M rows"
    with pytest.raises(ValueError):
        M.supersede(store, old["id"])


def test_anchored_memory_goes_stale_when_file_changes(proj):
    d, store = proj
    m = M.remember(store, "k_el fitted at 0.2 per hour", "fact", cwd=str(d), anchors=[str(d / "model.py")])
    assert "stale" not in M.get(store, m["id"])
    (d / "model.py").write_text("k_el = 0.35\n")
    assert M.get(store, m["id"])["stale"] == [f"anchor changed: {d / 'model.py'}"]
    (d / "model.py").unlink()
    assert M.get(store, m["id"])["stale"][0].startswith("anchor missing")


def test_brief_respects_budget_and_orders_sections(proj):
    d, store = proj
    M.remember(store, "Never retry the PBPK fit with scipy minimize; it diverges", "rejected", cwd=str(d))
    M.remember(store, "Fit in log-space", "decision", cwd=str(d))
    for i in range(30):
        M.remember(store, f"todo item number {i} " + "x" * 80, "todo", cwd=str(d))
    b = M.brief(store, cwd=str(d), max_chars=600)
    assert len(b["text"]) <= 600
    assert b["text"].index("Decisions") < b["text"].index("Rejected")
    assert "Fit in log-space" in b["text"] and "scipy minimize" in b["text"]
    assert M.brief(store, cwd=str(d.parent / "elsewhere"))["text"] == ""


def test_directives():
    text = "ok so\ndecision: use pgvector for team mode\n- rejected: celery, too heavy\nTODO: add retries\nnot: this"
    assert M.directives(text) == [("decision", "use pgvector for team mode"), ("rejected", "celery, too heavy"),
                                  ("todo", "add retries")]


def _claude_transcript(path, cwd):
    lines = [
        {"type": "user", "message": {"role": "user", "content": "Refit the clearance model\ndecision: fit in log-space"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Done."},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": str(cwd / "model.py")}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))
    return path


def test_claude_code_session_capture_end_to_end(proj, tmp_path):
    d, store = proj
    tr = _claude_transcript(tmp_path / "t.jsonl", d)
    start = H.handle("claude-code", {"hook_event_name": "SessionStart", "cwd": str(d), "session_id": "s1"})
    assert start == {}                                    # nothing remembered yet -> no context
    H.handle("claude-code", {"hook_event_name": "Stop", "cwd": str(d), "session_id": "s1", "transcript_path": str(tr)})
    H.handle("claude-code", {"hook_event_name": "Stop", "cwd": str(d), "session_id": "s1", "transcript_path": str(tr)})
    sessions = M.recall(store, "", cwd=str(d), kinds=["session"])["memories"]
    assert len(sessions) == 1                             # one per session, updated in place
    s = sessions[0]
    assert "goal: Refit the clearance model" in s["text"] and "model.py" in s["text"] and "ran: pytest" in s["text"]
    assert s["anchors"] == [str(d / "model.py")]
    decisions = M.recall(store, "", cwd=str(d), kinds=["decision"])["memories"]
    assert [x["text"] for x in decisions] == ["fit in log-space"] and decisions[0]["source"] == "claude-code"
    out = H.handle("claude-code", {"hook_event_name": "SessionStart", "cwd": str(d), "session_id": "s2"})
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "fit in log-space" in ctx and "Recent sessions" in ctx


def test_cursor_events_accumulate(proj):
    d, store = proj
    base = {"conversation_id": "c1", "workspace_roots": [str(d)]}
    H.handle("cursor", {**base, "hook_event_name": "beforeSubmitPrompt", "prompt": "add a dose-response plot"})
    H.handle("cursor", {**base, "hook_event_name": "afterFileEdit", "file_path": str(d / "plot.py")})
    H.handle("cursor", {**base, "hook_event_name": "afterShellExecution", "command": "python plot.py"})
    H.handle("cursor", {**base, "hook_event_name": "stop", "status": "completed"})
    (s,) = M.recall(store, "", cwd=str(d), kinds=["session"])["memories"]
    assert "add a dose-response plot" in s["text"] and "plot.py" in s["text"] and "ran: python" in s["text"]
    M.remember(store, "Plots use the viridis palette", "preference", cwd=str(d))
    out = H.handle("cursor", {**base, "hook_event_name": "sessionStart"})
    assert "viridis" in out["additional_context"]


def test_codex_notify(proj):
    d, store = proj
    H.handle("codex", {"type": "agent-turn-complete", "thread-id": "t1", "cwd": str(d),
                       "input-messages": ["speed up the index build"], "last-assistant-message": "todo: try fp16"})
    assert M.recall(store, "", cwd=str(d), kinds=["todo"])["memories"][0]["text"] == "try fp16"


def test_run_hook_never_raises():
    assert H.run_hook("cursor", "not json") == "{}"
    assert H.run_hook("cursor", "[1,2]") == "{}"


def test_installers_are_idempotent_and_preserve_other_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(H.Path, "home", lambda: tmp_path)
    cur = tmp_path / ".cursor" / "hooks.json"
    cur.parent.mkdir()
    cur.write_text(json.dumps({"version": 1, "hooks": {"postToolUse": [{"command": "git-ai checkpoint"}]}}))
    r = H.install("cursor")
    assert r["changed"] and "sessionStart" in r["events"]
    assert not H.install("cursor")["changed"]
    data = json.loads(cur.read_text())
    assert data["hooks"]["postToolUse"][0]["command"] == "git-ai checkpoint"
    assert (cur.parent / "hooks.json.bak-cartograph").exists()
    assert H.installed("cursor")
    H.uninstall("cursor")
    assert json.loads(cur.read_text())["hooks"] == {"postToolUse": [{"command": "git-ai checkpoint"}]}

    H.install("claude-code")
    cc = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert cc["hooks"]["SessionStart"][0]["matcher"] and H.installed("claude-code")

    codex = tmp_path / ".codex" / "config.toml"
    codex.parent.mkdir()
    codex.write_text('model = "o4"\n[mcp_servers.x]\ncommand = "x"\n')
    assert H.install("codex")["changed"] and codex.read_text().startswith("notify = [")
    assert not H.install("codex")["changed"]
    H.uninstall("codex")
    assert codex.read_text().startswith('model = "o4"')
    codex.write_text('notify = ["my-notifier"]\n')
    assert not H.install("codex")["changed"]             # never clobbers a user's own notify program


def test_mcp_memory_tools(proj):
    d, _ = proj
    call = lambda name, args: json.loads(_handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",  # noqa: E731
                                                   "params": {"name": name, "arguments": args}}))["result"]
    r = call("remember", {"text": "Cite tissue IDs in every figure caption", "kind": "preference", "cwd": str(d)})
    mid = json.loads(r["content"][0]["text"])["id"]
    r = call("recall", {"query": "figure caption", "cwd": str(d)})
    assert json.loads(r["content"][0]["text"])["memories"][0]["id"] == mid
    r = call("brief", {"cwd": str(d)})
    assert "tissue IDs" in json.loads(r["content"][0]["text"])["text"]
    r = call("supersede", {"id": mid, "reason": "journal style changed"})
    assert json.loads(r["content"][0]["text"])["valid_to"]
    assert call("remember", {"text": "x", "kind": "bogus"})["isError"] is True
    tools = {t["name"]: t for t in json.loads(_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))["result"]["tools"]}
    assert tools["remember"]["annotations"]["readOnlyHint"] is False
    assert tools["recall"]["annotations"]["readOnlyHint"] is True
