"""agent-setup output + token-budgeted steering brief."""

from typer.testing import CliRunner

from cartograph.cli import app
from cartograph.config import Config
from cartograph.demo import write_corpus
from cartograph.ingest import ingest_path
from cartograph.persona import build_brief
from cartograph.persona.profile import load_persona
from cartograph.storage import Store

runner = CliRunner()


def test_agent_setup_prints_config_and_rule():
    r = runner.invoke(app, ["agent-setup"])
    assert r.exit_code == 0
    assert "mcpServers" in r.stdout and "cartograph" in r.stdout
    assert "personalize" in r.stdout and "record_use" in r.stdout


def test_brief_respects_char_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    store = Store(tmp_path / "home" / "graph.sqlite")
    ingest_path(write_corpus(tmp_path / "corpus"), store, Config())
    p = load_persona(store)
    big = build_brief("validate input and handle errors", store, Config(), p, top_k=6, max_chars=0)
    small = build_brief("validate input and handle errors", store, Config(), p, top_k=6, max_chars=300)
    big_chars = sum(len(c["snippet"]) for c in big["relevant_context"])
    small_chars = sum(len(c["snippet"]) for c in small["relevant_context"])
    assert small_chars <= 300 or small_chars < big_chars   # budget actually constrains
