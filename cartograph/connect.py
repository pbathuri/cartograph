"""`carto connect` — register Cartograph's MCP server with the AI clients on this machine.

Every client keeps MCP servers in a different file and shape; this module knows them so users don't have
to. Edits are minimal (only our entry), idempotent, and the previous file is kept as `*.bak-cartograph`.
Files with comments (JSONC) are never rewritten — the snippet to paste is returned instead.

The server is launched with this interpreter (`python -m cartograph mcp-server`) rather than `carto`, so
GUI apps whose PATH lacks conda/venv/pipx still find it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

NAME = "cartograph"


def _appdata() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _zed() -> Path:
    if os.name == "nt":
        return _appdata() / "Zed" / "settings.json"
    return Path.home() / ".config" / "zed" / "settings.json"


# client -> (label, config path, format). format: json key holding servers, or "codex" / "continue".
CLIENTS: dict[str, tuple[str, Callable[[], Path], str]] = {
    "claude-code": ("Claude Code", lambda: Path.home() / ".claude.json", "mcpServers"),
    "claude-desktop": ("Claude Desktop", lambda: _appdata() / "Claude" / "claude_desktop_config.json", "mcpServers"),
    "cursor": ("Cursor", lambda: Path.home() / ".cursor" / "mcp.json", "mcpServers"),
    "windsurf": ("Windsurf", lambda: Path.home() / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers"),
    "vscode": ("VS Code (Copilot agent mode)", lambda: _appdata() / "Code" / "User" / "mcp.json", "servers"),
    "cline": ("Cline", lambda: _appdata() / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev"
              / "settings" / "cline_mcp_settings.json", "mcpServers"),
    "gemini": ("Gemini CLI", lambda: Path.home() / ".gemini" / "settings.json", "mcpServers"),
    "codex": ("OpenAI Codex", lambda: Path.home() / ".codex" / "config.toml", "codex"),
    "zed": ("Zed", _zed, "context_servers"),
    "continue": ("Continue", lambda: Path.home() / ".continue" / "mcpServers" / f"{NAME}.yaml", "continue"),
    "lm-studio": ("LM Studio", lambda: Path.home() / ".lmstudio" / "mcp.json", "mcpServers"),
    "goose": ("Goose", lambda: Path.home() / ".config" / "goose" / "config.yaml", "goose"),
}


def server(read_only: bool = False) -> dict:
    spec: dict = {"command": sys.executable, "args": ["-m", "cartograph", "mcp-server"] + (["--read-only"] if read_only else [])}
    if os.environ.get("CARTOGRAPH_HOME"):
        spec["env"] = {"CARTOGRAPH_HOME": os.environ["CARTOGRAPH_HOME"]}
    return spec


def _entry(client: str, spec: dict) -> dict:
    return {"type": "stdio", **spec} if client in ("vscode", "claude-code") else dict(spec)


def config_path(client: str) -> Path:
    return CLIENTS[client][1]()


def detected(client: str) -> bool:
    """The client looks installed: its config file, or the folder it lives in, exists."""
    p = config_path(client)
    if CLIENTS[client][2] == "continue":
        return p.parent.parent.exists()
    return p.exists() or p.parent.exists()


def _backup(p: Path) -> None:
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".bak-cartograph"))


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    _backup(p)
    tmp = p.with_name(p.name + ".tmp-cartograph")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _toml_block(spec: dict) -> str:
    lines = [f"[mcp_servers.{NAME}]", f"command = {json.dumps(spec['command'])}", f"args = {json.dumps(spec['args'])}"]
    if spec.get("env"):
        lines.append("env = { " + ", ".join(f"{k} = {json.dumps(v)}" for k, v in spec["env"].items()) + " }")
    return "\n".join(lines) + "\n"


_TOML_SECTION = re.compile(rf"(?ms)^\[mcp_servers\.{NAME}\]\s*\n.*?(?=^\[|\Z)")


def _yaml_continue(spec: dict) -> str:
    env = "".join(f"\n    env:\n      {k}: {json.dumps(v)}" for k, v in (spec.get("env") or {}).items())
    return (f"name: Cartograph\nversion: 2.0.0\nschema: v1\nmcpServers:\n  - name: {NAME}\n"
            f"    command: {json.dumps(spec['command'])}\n    args: {json.dumps(spec['args'])}{env}\n")


def snippet(client: str, read_only: bool = False) -> str:
    """What to paste by hand into the client's config."""
    spec, fmt = server(read_only), CLIENTS[client][2]
    if fmt == "codex":
        return _toml_block(spec)
    if fmt == "continue":
        return _yaml_continue(spec)
    if fmt == "goose":
        args = " ".join(spec["args"])
        return (f"extensions:\n  {NAME}:\n    name: {NAME}\n    type: stdio\n    enabled: true\n"
                f"    cmd: {json.dumps(spec['command'])}\n    args: {json.dumps(spec['args'])}\n    timeout: 300\n"
                f"# or: goose configure -> Add Extension -> Command-line: {spec['command']} {args}\n")
    return json.dumps({fmt: {NAME: _entry(client, spec)}}, indent=2)


def connect(client: str, *, read_only: bool = False, dry_run: bool = False) -> dict:
    """Add (or update) Cartograph in one client's MCP config. Idempotent."""
    if client not in CLIENTS:
        raise ValueError(f"unknown client {client!r}; one of: {', '.join(CLIENTS)}")
    label, _, fmt = CLIENTS[client]
    p, spec = config_path(client), server(read_only)
    out = {"client": client, "label": label, "path": str(p), "changed": False}
    if fmt == "goose":
        return {**out, "note": "add it by hand (Goose's YAML is hand-edited)", "snippet": snippet(client, read_only)}
    if fmt == "continue":
        new = _yaml_continue(spec)
        if p.exists() and p.read_text(encoding="utf-8") == new:
            return {**out, "note": "already connected"}
        if not dry_run:
            _write(p, new)
        return {**out, "changed": True}
    if fmt == "codex":
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        block = _toml_block(spec)
        m = _TOML_SECTION.search(text)
        if m and m.group(0).strip() == block.strip():
            return {**out, "note": "already connected"}
        new = (text[:m.start()] + block + ("\n" if text[m.end():].strip() else "") + text[m.end():]) if m \
            else (text.rstrip() + "\n\n" if text.strip() else "") + block
        if not dry_run:
            _write(p, new)
        return {**out, "changed": True, "note": "updated" if m else ""}
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError:
        return {**out, "note": "config has comments or isn't plain JSON; not rewriting it — paste this in by hand",
                "snippet": snippet(client, read_only)}
    if not isinstance(data, dict):
        return {**out, "note": "unexpected config shape; paste this in by hand", "snippet": snippet(client, read_only)}
    servers = data.setdefault(fmt, {})
    entry = _entry(client, spec)
    if servers.get(NAME) == entry:
        return {**out, "note": "already connected"}
    note = "updated" if NAME in servers else ""
    servers[NAME] = entry
    if not dry_run:
        _write(p, json.dumps(data, indent=2) + "\n")
    return {**out, "changed": True, "note": note}


def disconnect(client: str) -> dict:
    label, _, fmt = CLIENTS[client]
    p = config_path(client)
    out = {"client": client, "label": label, "path": str(p), "changed": False}
    if not p.exists() or fmt == "goose":
        return out
    if fmt == "continue":
        _backup(p)
        p.unlink()
        return {**out, "changed": True}
    text = p.read_text(encoding="utf-8")
    if fmt == "codex":
        new = _TOML_SECTION.sub("", text)
        if new != text:
            _write(p, new.rstrip() + "\n")
            return {**out, "changed": True}
        return out
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError:
        return {**out, "note": "config isn't plain JSON; remove the cartograph entry by hand"}
    if NAME in (data.get(fmt) or {}):
        del data[fmt][NAME]
        _write(p, json.dumps(data, indent=2) + "\n")
        return {**out, "changed": True}
    return out


def connected(client: str) -> bool:
    p, fmt = config_path(client), CLIENTS[client][2]
    try:
        if not p.exists():
            return False
        text = p.read_text(encoding="utf-8")
        if fmt == "continue":
            return True
        if fmt == "codex":
            return bool(_TOML_SECTION.search(text))
        if fmt == "goose":
            return re.search(rf"(?m)^\s+{NAME}:\s*$", text) is not None
        return NAME in (json.loads(text).get(fmt) or {})
    except (OSError, ValueError, AttributeError):
        return False


def status() -> list[dict]:
    return [{"client": c, "label": CLIENTS[c][0], "path": str(config_path(c)), "detected": detected(c),
             "connected": connected(c)} for c in CLIENTS]
