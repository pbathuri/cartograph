"""`elevate` — one-shot 'take this task to top-of-field' briefing for ANY field. Combines: the elite
bar (DoD), canonical reference repos, the frontier playbook (process), adjoining-field moves, and the
most relevant repos already in YOUR graph (hybrid retrieval). The single call an agent makes at task
start to understand what frontier-grade looks like here and what you already have to build on."""
from __future__ import annotations

from .catalog import elite_refs, match_field
from .dod import dod_for, score_build
from .playbooks import playbook_for


def elevate(task: str, store, cfg, *, project=None) -> dict:
    fld = match_field(task) or "general"
    dod = dod_for(task)
    out = {
        "task": task,
        "field": fld,
        "elite_bar": [f"{c['dim']}: {c['req']}" for c in dod["criteria"] if c["tier"] == "elite"][:6],
        "references": [f"{r['repo']} ({r['teaches']})" for r in elite_refs(task)["references"][:6]],
        "playbook": playbook_for(task)["steps"],
        "adjoining": elite_refs(task)["adjoining"],
        "relevant_existing": [],
        "current_grade": "n/a",
        "note": "elite practices need eval/human confirmation; references are public OSS to clone+ingest",
    }
    try:
        from ..retrieve import retrieve
        out["relevant_existing"] = retrieve(task, store, cfg, top_k=6).projects
    except Exception:
        pass
    if project is not None:
        try:
            out["current_grade"] = score_build(project, task)["grade"]
        except Exception:
            pass
    return out
