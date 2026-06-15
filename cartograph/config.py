"""Configuration + workspace paths. Everything lives under ~/.cartograph by default (override with
the CARTOGRAPH_HOME env var), so the package itself stays data-free and the user's graph is theirs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def home() -> Path:
    """The workspace root. Override with CARTOGRAPH_HOME to put your graph on a big/fast drive."""
    p = os.environ.get("CARTOGRAPH_HOME", "").strip()
    return Path(p) if p else Path.home() / ".cartograph"


def db_path() -> Path:
    return home() / "graph.sqlite"


def index_dir() -> Path:
    return home() / "index"


def config_path() -> Path:
    return home() / "config.yaml"


@dataclass
class Config:
    roots: list[str] = field(default_factory=list)         # folders to ingest (your repos, notes, docs)
    field_focus: list[str] = field(default_factory=list)   # your domains, e.g. ["ml_experiment", "web_frontend"]
    ignore: list[str] = field(default_factory=lambda: [
        ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
        ".next", "target", ".mypy_cache", ".pytest_cache", "site-packages",
    ])
    max_file_mb: float = 2.0
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5"

    def to_dict(self) -> dict[str, Any]:
        return {"roots": self.roots, "field_focus": self.field_focus, "ignore": self.ignore,
                "max_file_mb": self.max_file_mb, "embed_model": self.embed_model}


def load_config() -> Config:
    p = config_path()
    if not p.exists():
        return Config()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    c = Config()
    for k, v in data.items():
        if hasattr(c, k) and v is not None:
            setattr(c, k, v)
    return c


def save_config(c: Config) -> Path:
    home().mkdir(parents=True, exist_ok=True)
    config_path().write_text(yaml.safe_dump(c.to_dict(), sort_keys=False), encoding="utf-8")
    return config_path()
