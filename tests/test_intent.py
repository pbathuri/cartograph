"""Prompt-intent classifier + its effect on the steering brief."""
import pytest

from cartograph.config import Config
from cartograph.demo import write_corpus
from cartograph.ingest import ingest_path
from cartograph.intent import classify
from cartograph.persona import build_brief
from cartograph.persona.profile import load_persona
from cartograph.storage import Store


@pytest.mark.parametrize("prompt,expected", [
    ("why does this throw a TypeError exception", "debug"),
    ("implement a function to parse CSV", "code"),
    ("review my auth module and make it better", "review"),
    ("write a blog post about caching", "generate"),
    ("explain how reciprocal rank fusion works", "explain"),
    ("what's the capital of France", "general"),
])
def test_intent_classification(prompt, expected):
    assert classify(prompt)["intent"] == expected


def test_brief_includes_intent_and_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path / "home"))
    store = Store(tmp_path / "home" / "graph.sqlite")
    ingest_path(write_corpus(tmp_path / "corpus"), store, Config())
    b = build_brief("fix the bug where validation crashes on null", store, Config(), load_persona(store))
    assert b["prompt_intent"] == "debug"
    assert any("root cause" in g.lower() for g in b["output_guidance"])   # debug guidance present
