"""Tuned parameters — the knobs `carto improve` learns, read by retrieval and the steering brief.

Stored in ~/.cartograph/tuned.json. Every value has a production default, so a missing/old file is safe.
History of every tuning run (params, train/holdout metrics, accepted or not) is appended to
tuning_log.jsonl so a regression can be traced and rolled back (`carto improve --rollback`).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import home

DEFAULTS: dict = {
    "rrf_k": 60,               # Reciprocal Rank Fusion constant (lower = trust the top ranks more)
    "w_ppr": 0.0,              # weight of personalized-PageRank project score in re-ranking (0 = off)
    "ppr_damping": 0.85,
    "prior_mix": 0.3,          # share of PPR teleport mass from recent AutoLog activity
    "gate_div": 2.0,           # persona relevance gate: influence the top ~gate_div distinct projects
    "snips_per_project": 2,    # brief: max snippets from one project (diversity + token cost)
    "snippet_chars": 400,      # brief: chars per snippet
    "hint_budget": 3,          # brief: max frontier/method hints
}


def _path() -> Path:
    return home() / "tuned.json"


def _log_path() -> Path:
    return home() / "tuning_log.jsonl"


_CACHE: tuple[float, dict] | None = None


def load() -> dict:
    global _CACHE
    p = _path()
    try:
        m = p.stat().st_mtime
    except OSError:
        return dict(DEFAULTS)
    if _CACHE and _CACHE[0] == m:
        return dict(_CACHE[1])
    try:
        data = {**DEFAULTS, **{k: v for k, v in json.loads(p.read_text()).get("params", {}).items() if k in DEFAULTS}}
    except Exception:
        data = dict(DEFAULTS)
    _CACHE = (m, data)
    return dict(data)


def save(params: dict, metrics: dict) -> Path:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = None
    if p.exists():
        try:
            prev = json.loads(p.read_text())
        except Exception:
            prev = None
    body = {"params": {k: params[k] for k in DEFAULTS if k in params}, "metrics": metrics,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "previous": (prev or {}).get("params")}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2))
    os.replace(tmp, p)
    return p


def log_run(entry: dict) -> None:
    with _log_path().open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")


def rollback() -> dict | None:
    """Restore the parameters that were live before the last accepted tuning."""
    p = _path()
    if not p.exists():
        return None
    body = json.loads(p.read_text())
    prev = body.get("previous")
    if not prev:
        return None
    save(prev, {"rolled_back_from": body.get("params")})
    return prev
