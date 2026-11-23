"""`carto connect`: per-client MCP config edits are minimal, idempotent, backed up and reversible."""
import json
import sys

import pytest

from cartograph import connect as C


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("CARTOGRAPH_HOME", raising=False)
    return tmp_path


JSON_CLIENTS = [c for c, (_, _, fmt) in C.CLIENTS.items() if fmt not in ("codex", "continue", "goose")]


@pytest.mark.parametrize("client", JSON_CLIENTS)
def test_json_clients_keep_other_servers_and_are_idempotent(home, client):
    p = C.config_path(client)
    key = C.CLIENTS[client][2]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({key: {"other": {"command": "x"}}, "theme": "dark"}))
    r = C.connect(client)
    assert r["changed"]
    data = json.loads(p.read_text())
    assert data["theme"] == "dark" and data[key]["other"] == {"command": "x"}
    entry = data[key]["cartograph"]
    assert entry["command"] == sys.executable and entry["args"] == ["-m", "cartograph", "mcp-server"]
    assert p.with_name(p.name + ".bak-cartograph").exists()
    assert C.connected(client)
    assert not C.connect(client)["changed"]
    assert C.connect(client, read_only=True)["note"] == "updated"
    assert json.loads(p.read_text())[key]["cartograph"]["args"][-1] == "--read-only"
    assert C.disconnect(client)["changed"]
    assert "cartograph" not in json.loads(p.read_text())[key] and not C.connected(client)


def test_vscode_entry_has_stdio_type(home):
    C.connect("vscode")
    assert json.loads(C.config_path("vscode").read_text())["servers"]["cartograph"]["type"] == "stdio"


def test_jsonc_is_never_rewritten(home):
    p = C.config_path("zed")
    p.parent.mkdir(parents=True)
    original = '{\n  // my font\n  "buffer_font_size": 15,\n}\n'
    p.write_text(original)
    r = C.connect("zed")
    assert not r["changed"] and '"context_servers"' in r["snippet"]
    assert p.read_text() == original


def test_codex_toml_section_added_replaced_removed(home):
    p = C.config_path("codex")
    p.parent.mkdir(parents=True)
    p.write_text('model = "o4"\n\n[mcp_servers.other]\ncommand = "y"\n')
    assert C.connect("codex")["changed"]
    text = p.read_text()
    assert 'model = "o4"' in text and "[mcp_servers.other]" in text and "[mcp_servers.cartograph]" in text
    assert not C.connect("codex")["changed"]
    C.connect("codex", read_only=True)
    text = p.read_text()
    assert text.count("[mcp_servers.cartograph]") == 1 and "--read-only" in text
    assert C.disconnect("codex")["changed"]
    text = p.read_text()
    assert "cartograph" not in text and "[mcp_servers.other]" in text


def test_continue_block_file(home):
    (home / ".continue").mkdir()
    assert C.detected("continue")
    assert C.connect("continue")["changed"]
    text = C.config_path("continue").read_text()
    assert "mcpServers:" in text and "mcp-server" in text
    assert C.disconnect("continue")["changed"] and not C.config_path("continue").exists()


def test_env_is_passed_when_home_is_custom(home, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(home / "graph"))
    C.connect("cursor")
    entry = json.loads(C.config_path("cursor").read_text())["mcpServers"]["cartograph"]
    assert entry["env"] == {"CARTOGRAPH_HOME": str(home / "graph")}


def test_dry_run_and_status(home):
    (home / ".cursor").mkdir()
    r = C.connect("cursor", dry_run=True)
    assert r["changed"] and not C.config_path("cursor").exists()
    st = {s["client"]: s for s in C.status()}
    assert st["cursor"]["detected"] and not st["cursor"]["connected"]
    assert not st["windsurf"]["detected"]
    assert not st["claude-code"]["detected"] or C.shutil.which("claude")
    (home / ".claude").mkdir()
    assert C.detected("claude-code")
    with pytest.raises(ValueError):
        C.connect("nope")
