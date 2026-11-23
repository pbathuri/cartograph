"""Deterministic entities: precise extraction, evidence + temporal bookkeeping, entity-based project links."""
import json
import sqlite3
import time

from cartograph import entities as E
from cartograph import ingest as ing
from cartograph.config import Config, db_path
from cartograph.storage import Store


def names(facts, rel=None):
    return {f[1] for f in facts if rel is None or f[2] == rel}


def test_pyproject_reads_dependency_arrays_only():
    text = '''[project]
name = "cartograph__v1"
keywords = ["rag", "knowledge-graph"]
requires-python = ">=3.10"
dependencies = [
    "typer>=0.9",
    "PyYAML",
]
[project.optional-dependencies]
docs = ["pypdf>=4.0"]
[build-system]
build-backend = "setuptools.build_meta"
[tool.poetry.dependencies]
python = "^3.10"
httpx = "^0.27"
'''
    got = names(E.extract(text, "pyproject.toml"), "DEPENDS_ON")
    assert got == {"pypi/typer", "pypi/pyyaml", "pypi/pypdf", "pypi/httpx"}


def test_requirements_and_package_json_and_cargo_and_go():
    assert names(E.extract("numpy>=1.24\n# c\n-e .\nscikit_learn[all]==1.3\ntorch", "requirements-dev.txt")) == {
        "pypi/numpy", "pypi/scikit-learn", "pypi/torch"}
    pkg = '{"name": "app", "version": "1.0.0", "scripts": {"test": "jest"}, "engines": {"node": ">=18"},' \
          ' "dependencies": {"react": "^18.2.0", "@tanstack/query": "5.0.0"}}'
    assert names(E.extract(pkg, "package.json")) == {"npm/react", "npm/@tanstack/query"}
    cargo = '[package]\nname = "x"\nedition = "2021"\n[dependencies]\nserde = "1.0"\ntokio = { version = "1" }\n'
    assert names(E.extract(cargo, "Cargo.toml")) == {"cargo/serde", "cargo/tokio"}
    gomod = "module x\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n"
    assert names(E.extract(gomod, "go.mod"), "DEPENDS_ON") == {"go/github.com/gin-gonic/gin"}


def test_imports_skip_stdlib_relative_and_local_modules():
    py = "import os\nimport numpy as np\nfrom sklearn.metrics import f1\nfrom . import x\nfrom utils import y\n"
    assert names(E.extract(py, "a.py", frozenset({"utils"}))) == {"pypi/numpy", "pypi/scikit-learn"}
    js = "import React from 'react'\nimport fs from 'fs'\nimport x from './x'\nconst q = require('@scope/pkg/sub')"
    assert names(E.extract(js, "a.ts")) == {"npm/react", "npm/@scope/pkg"}


def test_identifiers():
    text = ("See doi 10.1016/j.ijpharm.2020.119585. and arXiv:1706.03762v5 or https://arxiv.org/abs/2005.11401; "
            "code at https://github.com/huggingface/transformers.git and github.com/orgs/foo; "
            "weights huggingface.co/meta-llama/Llama-3-8B, data huggingface.co/datasets/squad/v2")
    f = E.extract(text, "notes.md")
    assert names(f, "CITES") == {"doi:10.1016/j.ijpharm.2020.119585", "arxiv:1706.03762", "arxiv:2005.11401"}
    assert names(f, "REFERENCES") == {"github/huggingface/transformers"}
    assert names(f, "USES_MODEL") == {"hf/meta-llama/llama-3-8b"}
    assert names(f, "USES_DATASET") == {"hf/squad/v2"}


def _proj(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir()
    (d / ".git").mkdir()
    for fn, body in files.items():
        (d / fn).write_text(body)
    return d


def test_ingest_records_evidence_and_links_projects(tmp_path):
    paper = "Builds on arXiv:2401.01234 and doi 10.1167/iovs.61.8.1 for corneal transport.\n"
    a = _proj(tmp_path, "alpha", {"notes.md": paper, "requirements.txt": "pydantic\n"})
    b = _proj(tmp_path, "beta", {"README.md": paper})
    _proj(tmp_path, "gamma", {"README.md": "unrelated"})
    store = Store(db_path())
    st = ing.ingest_path(tmp_path, store, Config())
    assert st.facts > 0 and st.links >= 1
    facts = E.find(store, "2401.01234")
    assert {f["project"] for f in facts} == {"alpha", "beta"}
    f = facts[0]
    assert f["relation"] == "CITES" and f["confidence"] == 0.95 and f["extractor"] == "identifiers"
    assert f["evidence"]["cite"].endswith(":L1") and f["first_seen"]
    rel = E.related_projects(store, store.project_id_by_name("alpha"))
    assert rel[0]["project"] == "beta" and "paper:arxiv:2401.01234" in rel[0]["via"]
    dep = E.project_entities(store, store.project_id_by_name("alpha"), kind="package")
    assert dep[0]["entity"] == "package:pypi/pydantic" and dep[0]["relation"] == "DEPENDS_ON"
    assert a and b


def test_edits_keep_first_seen_and_drop_stale_facts(tmp_path, monkeypatch):
    d = _proj(tmp_path, "alpha", {"notes.md": "arXiv:2401.00001 and arXiv:2401.00002\n"})
    store = Store(db_path())
    ing.ingest_path(d, store, Config())
    first = E.find(store, "2401.00001")[0]["first_seen"]
    with store.cursor() as c:
        t0 = c.execute("SELECT min(first_seen) FROM mentions").fetchone()[0]
    monkeypatch.setattr(time, "time", lambda: t0 + 86400 * 3)
    (d / "notes.md").write_text("arXiv:2401.00001 only, now with more words\n")
    ing.ingest_path(d, store, Config())
    kept = E.find(store, "2401.00001")[0]
    assert kept["first_seen"] == first and kept["last_seen"] > first
    assert E.find(store, "2401.00002") == []


def test_backfill_matches_ingest_and_delete_project_cleans_up(tmp_path):
    d = _proj(tmp_path, "alpha", {"a.py": "import pandas\n", "n.md": "arXiv:2401.00003\n"})
    store = Store(db_path())
    ing.ingest_path(d, store, Config())
    before = E.counts(store)
    with store.cursor() as c:
        c.execute("DELETE FROM mentions")
    out = E.backfill(store)
    assert out["facts"] == before["facts"] and out["entities"] == before["entities"]
    store.delete_project(store.project_id_by_name("alpha"))
    assert E.counts(store)["facts"] == 0


def test_read_only_graph_without_entity_tables(tmp_path):
    db = tmp_path / "old.sqlite"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, path TEXT, field TEXT);"
                    "CREATE TABLE files (id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT);")
    c.close()
    ro = Store(db, read_only=True)
    assert E.find(ro, "x") == [] and E.counts(ro) == {"entities": 0, "facts": 0} and E.link_projects(ro) == 0


def test_mcp_entities_tool(tmp_path):
    from cartograph.mcp_server import _handle
    _proj(tmp_path, "alpha", {"n.md": "arXiv:2401.00004\n"})
    ing.ingest_path(tmp_path / "alpha", Store(db_path()), Config())
    r = json.loads(_handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "entities", "arguments": {"find": "2401.00004"}}}))
    facts = json.loads(r["result"]["content"][0]["text"])["facts"]
    assert facts[0]["project"] == "alpha" and facts[0]["evidence"]["cite"].endswith("n.md:L1")
    r = json.loads(_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "entities", "arguments": {}}}))
    assert r["result"]["isError"] is True
