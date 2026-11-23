"""MCP contract: schemas a strict client can validate, honest read/write annotations, read-only mode,
tool failures reported as isError results, and version negotiation."""
import json

from cartograph import mcp_server as m


def _call(method, params=None, rid=1):
    out = m._handle({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
    return json.loads(out)


def test_array_params_are_arrays():
    tools = {t["name"]: t for t in _call("tools/list")["result"]["tools"]}
    props = tools["record_use"]["inputSchema"]["properties"]
    assert props["helped"]["type"] == "array" and props["helped"]["items"] == {"type": "string"}
    assert tools["record_use"]["inputSchema"]["required"] == ["helped"]
    assert tools["retrieve_context"]["inputSchema"]["properties"]["top_k"]["type"] == "integer"


def test_annotations_separate_reads_from_writes():
    tools = {t["name"]: t for t in _call("tools/list")["result"]["tools"]}
    assert tools["record_use"]["annotations"]["readOnlyHint"] is False
    assert tools["retrieve_context"]["annotations"]["readOnlyHint"] is True
    assert all(t["annotations"]["destructiveHint"] is False for t in tools.values())


def test_read_only_mode_hides_and_refuses_writes(monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_MCP_READ_ONLY", "1")
    names = {t["name"] for t in _call("tools/list")["result"]["tools"]}
    assert "record_use" not in names and "retrieve_context" in names
    r = _call("tools/call", {"name": "record_use", "arguments": {"helped": ["x"]}})
    assert r["error"]["code"] == -32602


def test_tool_failure_is_an_iserror_result():
    from cartograph.config import db_path
    from cartograph.storage import Store
    Store(db_path())
    r = _call("tools/call", {"name": "retrieve_context", "arguments": {}})
    assert r["result"]["isError"] is True and "query is required" in r["result"]["content"][0]["text"]


def test_unknown_tool_is_a_protocol_error():
    assert _call("tools/call", {"name": "nope"})["error"]["code"] == -32602


def test_protocol_version_negotiation():
    assert _call("initialize", {"protocolVersion": "2025-03-26"})["result"]["protocolVersion"] == "2025-03-26"
    assert _call("initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == m.PROTOCOL_VERSIONS[0]


def test_lenient_array_arguments():
    assert m._names(["a", "b"]) == ["a", "b"]
    assert m._names('["a", "b"]') == ["a", "b"]
    assert m._names("a, b") == ["a", "b"]
    assert m._names(None) == []


def test_stale_import_heals(monkeypatch):
    calls = {"n": 0}
    real = m._run

    def flaky(rid, name, args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModuleNotFoundError("No module named 'cartograph.persona'")
        return real(rid, name, args)

    from cartograph.config import db_path
    from cartograph.storage import Store
    Store(db_path())
    monkeypatch.setattr(m, "_run", flaky)
    out = json.loads(m._dispatch(1, "graph_stats", {}))
    assert "result" in out and calls["n"] == 2


def test_missing_graph_explains_what_to_run():
    r = _call("tools/call", {"name": "graph_stats", "arguments": {}})
    assert r["result"]["isError"] and "carto ingest" in r["result"]["content"][0]["text"]
