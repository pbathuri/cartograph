"""Automatic session capture for AI coding agents, through each client's own hook system.

At session start the client receives Cartograph's memory brief for the project it opened. While the agent
works, Cartograph notes the goal, the files it edited and the commands it ran; at the end it writes one
`session` memory anchored to those files. Lines like `decision: use pgvector`, `rejected: celery`,
`todo: add retries` said in the chat become memories of that kind. No LLM is involved.

Supported: Claude Code, Cursor, Gemini CLI (hooks) and Codex (`notify`). Hooks never block the agent:
any error is swallowed and the hook prints an empty result."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .config import home

MARK = "cartograph memory hook"
CLIENTS = ("claude-code", "cursor", "gemini", "codex")
_EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit", "str_replace", "strreplace", "apply_patch",
               "create", "write_file", "replace", "edit_file"}
_SHELL_TOOLS = {"bash", "shell", "run_shell_command", "run_terminal_cmd", "exec_command"}


def command(client: str) -> list[str]:
    return [sys.executable, "-m", "cartograph", "memory", "hook", "--client", client]


def _cmd_str(client: str) -> str:
    return " ".join(f'"{a}"' if " " in a else a for a in command(client))


# ------------------------------------------------------------------------------------------ scratch
def _scratch_dir() -> Path:
    d = home() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scratch(client: str, sid: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid)[:120] or "default"
    return _scratch_dir() / f"{client}-{safe}.json"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(p: Path, d: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------ transcripts
def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") in ("text", "input_text", "output_text") and c.get("text"):
                out.append(c["text"])
            elif isinstance(c, str):
                out.append(c)
    return out


def _tool_uses(content: Any) -> list[tuple[str, dict]]:
    out = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") in ("tool_use", "tool_call", "function_call"):
                inp = c.get("input") or c.get("arguments") or {}
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except json.JSONDecodeError:
                        inp = {"raw": inp}
                out.append((str(c.get("name", "")), inp if isinstance(inp, dict) else {}))
    return out


_QUERY = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.S)
_NOISE = re.compile(r"<(system-reminder|timestamp|attached_files|system_notification|command-[\w-]+|"
                    r"local-command-[\w-]+)>.*?</\1>", re.S)
_SYNTHETIC = ("this session is being continued", "caveat:", "[request interrupted", "briefly inform the user",
              "<task-notification", "base directory for this skill")


def clean_prompt(text: str) -> str:
    """The human request inside a client's wrapped user message, or '' for system-generated ones."""
    m = _QUERY.search(text)
    t = m.group(1) if m else _NOISE.sub("", text)
    t = t.strip()
    if not t or t.lower().startswith(_SYNTHETIC) or t.startswith("<"):
        return ""
    return t


def parse_transcript(path: str, limit_bytes: int = 30_000_000) -> dict:
    """Prompts, assistant text, edited files and shell commands from a JSONL transcript (Claude Code,
    Cursor and Gemini write similar role/content records). Unknown lines are skipped."""
    out: dict = {"prompts": [], "assistant": [], "files": [], "commands": []}
    p = Path(path).expanduser()
    try:
        if not p.is_file() or p.stat().st_size > limit_bytes:
            return out
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else ev
        role = msg.get("role") or ev.get("type") or ev.get("role")
        content = msg.get("content")
        if role == "user":
            for t in _texts(content):
                t = clean_prompt(t)
                if t:
                    out["prompts"].append(t)
        elif role == "assistant":
            out["assistant"] += _texts(content)
            for name, inp in _tool_uses(content):
                _note_tool(out, name, inp)
    return out


def _note_tool(acc: dict, name: str, inp: dict) -> None:
    n = name.lower().replace("-", "_")
    if n in _EDIT_TOOLS or n.endswith(("edit", "write")):
        fp = inp.get("file_path") or inp.get("path") or inp.get("target_file") or inp.get("notebook_path")
        if fp and fp not in acc["files"]:
            acc["files"].append(str(fp))
    elif n in _SHELL_TOOLS:
        cmd = inp.get("command") or inp.get("cmd")
        if isinstance(cmd, list):
            cmd = " ".join(map(str, cmd))
        if cmd:
            acc["commands"].append(str(cmd)[:200])


# ------------------------------------------------------------------------------------------ handler
def _event(payload: dict) -> str:
    return str(payload.get("hook_event_name") or payload.get("hookEventName") or payload.get("type") or "").lower()


def _sid(payload: dict) -> str:
    return str(payload.get("session_id") or payload.get("conversation_id") or payload.get("thread-id")
               or payload.get("thread_id") or payload.get("sessionId") or "default")


def _cwd(payload: dict) -> str | None:
    roots = payload.get("workspace_roots")
    if isinstance(roots, list) and roots:
        return str(roots[0])
    return payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def _context_output(client: str, text: str) -> dict:
    if not text:
        return {}
    if client == "cursor":
        return {"additional_context": text}
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}


def summarize(client: str, acc: dict, cwd: str | None) -> str:
    goal = (acc.get("prompts") or [""])[0].strip().replace("\n", " ")
    files = acc.get("files") or []
    cmds = acc.get("commands") or []
    where = Path(cwd).name if cwd else "?"
    parts = [f"{client} session in {where}"]
    if goal:
        parts.append(f"goal: {goal[:240]}")
    if files:
        shown = ", ".join(Path(f).name for f in files[:8])
        parts.append(f"edited {len(files)} file(s): {shown}" + (f" (+{len(files) - 8})" if len(files) > 8 else ""))
    if cmds:
        heads = []
        for c in cmds:
            for seg in re.split(r"&&|\|\||[;|\n]", c):
                toks = [t for t in seg.split() if "=" not in t and t not in ("sudo", "env", "time", "nohup", "(", "{")]
                if not toks or toks[0] in ("cd", "export", "echo", "sleep", "#"):
                    continue
                h = Path(toks[0].strip("'\"()")).name
                if h and not h.startswith(("~", "$", "-")) and h not in heads:
                    heads.append(h)
        parts.append("ran: " + ", ".join(heads[:8]))
    if len(acc.get("prompts") or []) > 1:
        parts.append(f"{len(acc['prompts'])} prompts")
    return "; ".join(parts)


def handle(client: str, payload: dict, store=None) -> dict:
    """Process one hook event; returns the JSON the client expects on stdout."""
    from . import memory
    from .config import db_path
    from .storage import Store
    ev = _event(payload)
    cwd = _cwd(payload)
    if not db_path().exists():
        return {}
    if ev in ("sessionstart", "session_start"):
        store = store or Store(db_path(), read_only=True)
        return _context_output(client, memory.brief(store, cwd=cwd, max_chars=2000)["text"])
    sid = _sid(payload)
    sp = _scratch(client, sid)
    acc = _load(sp) or {"prompts": [], "assistant": [], "files": [], "commands": [], "cwd": cwd, "started": time.time()}
    if payload.get("prompt") and clean_prompt(str(payload["prompt"])):
        acc["prompts"].append(clean_prompt(str(payload["prompt"])))
    if payload.get("file_path"):
        if payload["file_path"] not in acc["files"]:
            acc["files"].append(str(payload["file_path"]))
    if payload.get("command") and ev.startswith(("aftershell", "beforeshell")):
        acc["commands"].append(str(payload["command"])[:200])
    if isinstance(payload.get("tool_input"), dict):
        _note_tool(acc, str(payload.get("tool_name", "")), payload["tool_input"])
    if client == "codex":                                  # notify: one call per finished turn
        acc["prompts"] += [m for m in payload.get("input-messages") or [] if isinstance(m, str)]
        if payload.get("last-assistant-message"):
            acc["assistant"].append(str(payload["last-assistant-message"]))
    final = ev in ("stop", "sessionend", "session_end", "agent-turn-complete")
    if payload.get("transcript_path") and final:
        t = parse_transcript(payload["transcript_path"])
        for k in ("prompts", "assistant", "commands"):
            acc[k] = t[k] or acc[k]
        acc["files"] = list(dict.fromkeys(acc["files"] + t["files"]))
    _save(sp, acc)
    if not final:
        return {}
    store = store or Store(db_path())
    if acc["prompts"] or acc["files"]:
        memory.upsert_session(store, client, sid, summarize(client, acc, acc.get("cwd") or cwd),
                              cwd=acc.get("cwd") or cwd, anchors=acc["files"])
    seen = set(acc.get("directives") or [])
    for kind, text in memory.directives("\n".join(acc["prompts"] + acc["assistant"])):
        key = f"{kind}:{text}"
        if key in seen:
            continue
        seen.add(key)
        memory.remember(store, text, kind, cwd=acc.get("cwd") or cwd, source=client, session=sid, confidence=0.8)
    acc["directives"] = sorted(seen)
    if ev in ("sessionend", "session_end"):
        sp.unlink(missing_ok=True)
    else:
        _save(sp, acc)
    return {}


def run_hook(client: str, raw: str) -> str:
    """Entry point used by `carto memory hook`: never raises, always prints valid JSON."""
    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
        return json.dumps(handle(client, payload))
    except Exception as e:  # noqa: BLE001 - a hook must never break the agent
        try:
            log = home() / "logs" / "hooks.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%F %T')} {client}: {type(e).__name__}: {e}\n")
        except OSError:
            pass
        return "{}"


# ------------------------------------------------------------------------------------------ installers
_CODEX_MARK = '"cartograph", "memory", "hook"'


def installed(client: str) -> bool:
    p = config_path(client)
    try:
        if not p.exists():
            return False
        if client == "codex":
            return _CODEX_MARK in p.read_text(encoding="utf-8")
        return any(_ours(e) for lst in (_read_json(p).get("hooks") or {}).values() for e in lst)
    except (OSError, ValueError):
        return False


def config_path(client: str) -> Path:
    h = Path.home()
    return {"claude-code": h / ".claude" / "settings.json", "cursor": h / ".cursor" / "hooks.json",
            "gemini": h / ".gemini" / "settings.json", "codex": h / ".codex" / "config.toml"}[client]


_EVENTS = {
    "claude-code": {"SessionStart": "startup|resume|clear|compact", "UserPromptSubmit": None, "Stop": None,
                    "SessionEnd": None},
    "gemini": {"SessionStart": None, "SessionEnd": None},
    "cursor": ["sessionStart", "beforeSubmitPrompt", "afterFileEdit", "afterShellExecution", "stop"],
}


def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8").strip()
    return json.loads(txt) if txt else {}


def _backup(p: Path) -> None:
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".bak-cartograph"))


def _ours(entry: Any) -> bool:
    return MARK.replace(" ", "") in json.dumps(entry).replace(" ", "").replace('\\"', "")


def install(client: str, *, dry_run: bool = False) -> dict:
    """Add Cartograph's hook entries to a client's user config, keeping everything else. Idempotent."""
    p = config_path(client)
    cmd = _cmd_str(client)
    if client == "codex":
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        if _CODEX_MARK in text:
            return {"client": client, "path": str(p), "changed": False, "note": "already installed"}
        if re.search(r"(?m)^\s*notify\s*=", text):
            return {"client": client, "path": str(p), "changed": False,
                    "note": "codex already has a `notify` program; add Cartograph to it by hand: " + json.dumps(command(client))}
        new = "notify = " + json.dumps(command(client)) + "\n" + text
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            _backup(p)
            p.write_text(new, encoding="utf-8")
        return {"client": client, "path": str(p), "changed": True, "events": ["notify"]}
    data = _read_json(p)
    hooks = data.setdefault("hooks", {})
    added = []
    if client == "cursor":
        data.setdefault("version", 1)
        for ev in _EVENTS["cursor"]:
            lst = hooks.setdefault(ev, [])
            if not any(_ours(e) for e in lst):
                lst.append({"command": cmd, "timeout": 10})
                added.append(ev)
    else:
        for ev, matcher in _EVENTS[client].items():
            lst = hooks.setdefault(ev, [])
            if not any(_ours(e) for e in lst):
                entry: dict = {"hooks": [{"type": "command", "command": cmd, "timeout": 10}]}
                if matcher:
                    entry["matcher"] = matcher
                lst.append(entry)
                added.append(ev)
    if added and not dry_run:
        p.parent.mkdir(parents=True, exist_ok=True)
        _backup(p)
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {"client": client, "path": str(p), "changed": bool(added), "events": added}


def uninstall(client: str) -> dict:
    p = config_path(client)
    if not p.exists():
        return {"client": client, "changed": False}
    if client == "codex":
        text = p.read_text(encoding="utf-8")
        new = "\n".join(line for line in text.splitlines()
                        if not (line.strip().startswith("notify") and _CODEX_MARK in line))
        if new.strip() != text.strip():
            _backup(p)
            p.write_text(new + "\n", encoding="utf-8")
            return {"client": client, "changed": True}
        return {"client": client, "changed": False}
    data = _read_json(p)
    removed = []
    for ev, lst in list((data.get("hooks") or {}).items()):
        keep = [e for e in lst if not _ours(e)]
        if len(keep) != len(lst):
            removed.append(ev)
            if keep:
                data["hooks"][ev] = keep
            else:
                del data["hooks"][ev]
    if removed:
        _backup(p)
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {"client": client, "changed": bool(removed), "events": removed}


def status() -> list[dict]:
    return [{"client": c, "config": str(config_path(c)), "exists": config_path(c).exists(), "installed": installed(c)}
            for c in CLIENTS]
