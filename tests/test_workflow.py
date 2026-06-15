"""Workflow overlay: locked core is immutable (returns a disclaimer), optional nodes disable, custom +
model nodes add/remove, edges respect locks, positions persist, and the merged graph validates."""
from cartograph import workflow as wf


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))


def test_default_workflow_has_locked_spine(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    g = wf.get_workflow()
    locked = {n["id"] for n in g["nodes"] if n.get("locked")}
    assert {"redact", "router", "persona", "retrieval", "brief"} <= locked
    assert wf.validate()["ok"]


def test_locked_node_delete_refused_with_disclaimer(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    r = wf.delete_node("redact")
    assert r["ok"] is False and r["locked"] is True and "privacy" in r["disclaimer"].lower()


def test_optional_node_disables_not_destroys(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    r = wf.delete_node("intent")                         # optional system node
    assert r["ok"] and r.get("disabled")
    assert any(n["id"] == "intent" and n["disabled"] for n in wf.get_workflow()["nodes"])
    wf.enable_node("intent")
    assert any(n["id"] == "intent" and not n["disabled"] for n in wf.get_workflow()["nodes"])


def test_import_model_is_hovering_and_removable(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    r = wf.import_model("my-rr", "./rr.pt", "model")
    nid = r["id"]
    assert r["hovering"]
    assert any(n["id"] == nid and n.get("hovering") for n in wf.get_workflow()["nodes"])
    assert wf.delete_node(nid)["removed"]                # custom/model nodes are freely removable
    assert not any(n["id"] == nid for n in wf.get_workflow()["nodes"])


def test_cannot_sever_locked_edge(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    r = wf.disconnect("reranker", "brief")               # part of the locked core path
    assert r["ok"] is False and r["locked"] is True


def test_custom_connect_and_position_persist(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    a = wf.add_node("note")["id"]
    wf.connect(a, "brief")
    wf.set_position(a, 333, 222)
    g = wf.get_workflow()
    assert any(e["from"] == a and e["to"] == "brief" for e in g["edges"])
    assert next(n for n in g["nodes"] if n["id"] == a)["pos"] == [333, 222]


def test_node_details_and_graph_menu(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    d = wf.node_details("persona")
    assert d["ok"] and d["locked"] and d["description"] and d["outgoing"]
    menu = wf.list_graphs()
    assert {g["id"] for g in menu} == {"cognitive", "pipeline", "vision"}
    assert all(g["connects"] for g in menu)
