from pathlib import Path

from cartograph.config import Config
from cartograph.ingest import discover_projects, ingest_path, project_name
from cartograph.storage import Store


def _w(p: Path, text: str = "def f():\n    return 1\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _cfg() -> Config:
    return Config(roots=[])


def test_clones_under_category_folders_become_separate_projects(tmp_path):
    root = tmp_path / "GitHub"
    for cat, repo in [("agent_app", "microsoft__autogen"), ("agent_app", "stanfordnlp__dspy"),
                      ("quant_research", "microsoft__qlib")]:
        _w(root / "References" / cat / repo / "pyproject.toml", "[project]\nname='x'\n")
        _w(root / "References" / cat / repo / "src" / "a.py")
    _w(root / "Quant_Practice" / "notes.md", "factor model notes")
    _w(root / "Quant_Practice" / "sub" / "b.py")
    found = {project_name(root, p): rec for p, rec in discover_projects(root, _cfg())}
    assert found == {
        "References/agent_app/microsoft__autogen": True,
        "References/agent_app/stanfordnlp__dspy": True,
        "References/quant_research/microsoft__qlib": True,
        "Quant_Practice": True,          # no markers inside -> stays one project
    }


def test_conventional_container_name_splits_unmarked_projects(tmp_path):
    root = tmp_path / "research"
    for name in ("02_Blink_MS2", "05_ROCK_MD"):
        _w(root / "Projects" / name / "analysis.py")
    _w(root / "Projects" / "README.md", "index of projects")
    _w(root / "Handoff" / "report.md", "handoff")
    found = {project_name(root, p): rec for p, rec in discover_projects(root, _cfg())}
    assert found == {"Projects": False,                  # container's own loose files, shallow
                     "Projects/02_Blink_MS2": True, "Projects/05_ROCK_MD": True, "Handoff": True}


def test_marked_root_is_one_project(tmp_path):
    _w(tmp_path / "repo" / ".git" / "HEAD", "ref")
    _w(tmp_path / "repo" / "a" / "x.py")
    _w(tmp_path / "repo" / "b" / "y.py")
    assert discover_projects(tmp_path / "repo", _cfg()) == [(tmp_path / "repo", True)]


def test_reingest_prunes_old_blob_and_shallow_project_skips_children(tmp_path):
    root = tmp_path / "GitHub"
    for repo in ("one", "two"):
        _w(root / "References" / repo / ".git" / "HEAD", "ref")
        _w(root / "References" / repo / "m.py", f"def {repo}():\n    pass\n")
    _w(root / "Other" / "n.md", "notes")
    store = Store(tmp_path / "g.sqlite")
    # simulate the old behaviour: the whole References folder stored as one project
    pid = store.upsert_project("References", str(root / "References"), "web_frontend")
    fid = store.upsert_file(pid, str(root / "References" / "one" / "m.py"), ".py", "h")
    store.add_chunks(fid, ["old blob chunk"])
    ingest_path(root, store, _cfg())
    with store.cursor() as c:
        names = sorted(r["name"] for r in c.execute("SELECT name FROM projects"))
        blob_hits = c.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'blob'").fetchone()[0]
    assert names == ["Other", "References/one", "References/two"]
    assert blob_hits == 0                                  # FTS rows removed with the stale project


def test_prune_outside_keeps_screen_activity_and_unmounted_roots(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    keep = store.upsert_project("mine", str(tmp_path / "roots" / "a" / "mine"), "ml_experiment")
    drive = store.upsert_project("on_drive", "/Volumes/Unplugged/x/proj", "hpc")
    store.upsert_project("stale", "/old/config/proj", "Finance")
    store.upsert_project("screen-activity", "", "screen-activity")
    n = store.prune_outside([str(tmp_path / "roots" / "a"), "/Volumes/Unplugged/x"], keep_names={"screen-activity"})
    with store.cursor() as c:
        names = sorted(r["name"] for r in c.execute("SELECT name FROM projects"))
    assert n == 1 and names == ["mine", "on_drive", "screen-activity"]
    assert keep and drive


def test_persona_ignores_reference_clones(tmp_path):
    from cartograph.persona.profile import build_from_corpus
    store = Store(tmp_path / "g.sqlite")
    for i in range(5):
        store.upsert_project(f"ref{i}", f"/u/GitHub/References/agent_app/r{i}", "agent_app")
    store.upsert_project("thesis", "/u/research/thesis", "quant_research")
    p = build_from_corpus(store)
    assert p.field_weights == {"quant_research": 1.0}
