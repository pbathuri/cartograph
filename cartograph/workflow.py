"""The editable workflow overlay — what the Studio app shows and lets non-technical users customize.

Two things are kept strictly separate:
  * the **cognitive graph** (projects/chunks in SQLite) — your DATA, never destructively edited here;
  * the **processing pipeline** — the ordered LAYERS your prompt flows through (capture -> novelty ->
    OCR -> redact -> router -> intent -> persona -> retrieval -> reranker -> brief). THIS is what the
    Studio edits, LangGraph-style: drop in your own model nodes between stages, disable optional stages,
    add context notes. The essential spine is LOCKED — attempting to remove it returns a disclaimer
    explaining exactly what would break, instead of letting the user silently break their setup.

The system spine is code (below). User edits live in a small overlay JSON in the workspace, applied on
top — so an upgrade that changes the spine never corrupts a user's customizations, and the core is
untouchable by design.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import home

# kind: source | io | transform | model | output   ·   locked spine nodes carry a `reason`.
_SPINE: list[dict] = [
    {"id": "corpus", "label": "Your corpus", "kind": "source", "locked": True,
     "reason": "Your ingested folders ARE the graph. Without them there is nothing to retrieve.",
     "desc": "Files you ingested (carto ingest): code, docs, notes — chunked + FTS/semantic indexed."},
    {"id": "vision_capture", "label": "Screen capture", "kind": "io", "optional": True,
     "desc": "Optional real-time screenshots (carto watch). Off unless you enable it; local-only."},
    {"id": "vision_novelty", "label": "Novelty cache", "kind": "transform", "optional": True,
     "desc": "Perceptual-hash + text-similarity gate. Skips near-duplicate frames so OCR runs rarely."},
    {"id": "vision_ocr", "label": "OCR", "kind": "transform", "optional": True,
     "desc": "Reads text from genuinely-new screens (Tesseract). Only runs on cache misses."},
    {"id": "redact", "label": "Redact secrets/PII", "kind": "transform", "locked": True,
     "reason": "Disabling redaction risks writing passwords, API keys, cards, and personal data into "
               "your graph in plain text. This is a privacy guardrail and cannot be removed.",
     "desc": "Strips secrets/PII from captured text BEFORE storage. Conservative by design."},
    {"id": "router", "label": "Field router", "kind": "model", "locked": True,
     "reason": "The router directs each prompt to the right field/context. Remove it and retrieval "
               "becomes field-blind — every answer loses your domain grounding.",
     "desc": "Learned per-field centroids route a prompt to its field (keyword fallback). carto train."},
    {"id": "intent", "label": "Intent classifier", "kind": "transform", "optional": True,
     "desc": "Detects use-case (debug/code/review/generate/explain) to shape output guidance."},
    {"id": "persona", "label": "Persona", "kind": "model", "locked": True,
     "reason": "The persona is the personalization core (field weights + preference subspaces + learned "
               "alpha). Without it, outputs are generic — this is the whole point of Cartograph.",
     "desc": "Per-field preference vectors + weights learned from your feedback and activity."},
    {"id": "retrieval", "label": "Hybrid retrieval", "kind": "transform", "locked": True,
     "reason": "Hybrid (semantic + keyword) retrieval is how your data reaches the answer. Removing it "
               "disconnects everything downstream from your corpus.",
     "desc": "Reciprocal-rank-fusion of semantic + FTS keyword search over your graph."},
    {"id": "reranker", "label": "Learned reranker", "kind": "model", "optional": True,
     "desc": "Logistic model trained on your feedback; replaces the heuristic blend when trained."},
    {"id": "brief", "label": "Steering brief", "kind": "output", "locked": True,
     "reason": "The brief is the model-agnostic output contract your agents consume. Removing it "
               "disconnects Cartograph from Claude/Cursor/ChatGPT/Gemini entirely.",
     "desc": "The compact envelope (persona + field + context + guidance + live activity) agents prepend."},
]
_SPINE_EDGES = [
    ("corpus", "retrieval"), ("vision_capture", "vision_novelty"), ("vision_novelty", "vision_ocr"),
    ("vision_ocr", "redact"), ("redact", "corpus"), ("router", "retrieval"),
    ("retrieval", "reranker"), ("reranker", "brief"), ("persona", "reranker"), ("persona", "brief"),
    ("intent", "brief"),
]
_LOCKED_EDGES = {("corpus", "retrieval"), ("router", "retrieval"), ("retrieval", "reranker"),
                 ("reranker", "brief"), ("persona", "brief"), ("redact", "corpus")}
_SPINE_IDS = {n["id"] for n in _SPINE}


def _overlay_path() -> Path:
    return home() / "workflow.json"


def load_overlay() -> dict:
    p = _overlay_path()
    base = {"nodes": [], "edges": [], "disabled": [], "removed": [], "positions": {}}
    if p.exists():
        try:
            base.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return base


def save_overlay(ov: dict) -> None:
    home().mkdir(parents=True, exist_ok=True)
    _overlay_path().write_text(json.dumps(ov, indent=2), encoding="utf-8")


def _default_pos(i: int) -> list[int]:
    return [120 + (i % 3) * 220, 80 + i * 70]


def get_workflow() -> dict:
    """Merged view the Studio renders: system spine (with lock flags) + user nodes, minus disabled."""
    ov = load_overlay()
    disabled, removed, pos = set(ov["disabled"]), set(ov["removed"]), ov["positions"]
    nodes = []
    for i, n in enumerate(_SPINE):
        node = dict(n)
        node["system"] = True
        node["locked"] = bool(n.get("locked"))
        node["disabled"] = n["id"] in disabled
        node["pos"] = pos.get(n["id"], _default_pos(i))
        nodes.append(node)
    for j, n in enumerate(ov["nodes"]):
        if n["id"] in removed:
            continue
        node = dict(n)
        node.update(system=False, locked=False, disabled=n["id"] in disabled,
                    pos=pos.get(n["id"], _default_pos(len(_SPINE) + j)))
        nodes.append(node)
    edges = [{"from": a, "to": b, "locked": (a, b) in _LOCKED_EDGES} for a, b in _SPINE_EDGES]
    edges += [{"from": e["from"], "to": e["to"], "locked": False} for e in ov["edges"]]
    alive = {n["id"] for n in nodes}
    edges = [e for e in edges if e["from"] in alive and e["to"] in alive]
    return {"nodes": nodes, "edges": edges}


def _disclaimer(node: dict) -> dict:
    return {"ok": False, "locked": True, "id": node["id"], "label": node["label"],
            "disclaimer": node.get("reason", "This is a core node and cannot be modified."),
            "hint": "You can still insert your OWN nodes around it, or disable optional nodes instead."}


def _find_spine(node_id: str) -> dict | None:
    return next((n for n in _SPINE if n["id"] == node_id), None)


def _next_id(prefix: str, ov: dict) -> str:
    n = 1 + sum(1 for x in ov["nodes"] if x["id"].startswith(prefix))
    while any(x["id"] == f"{prefix}{n}" for x in ov["nodes"]):
        n += 1
    return f"{prefix}{n}"


def add_node(label: str, kind: str = "custom", desc: str = "") -> dict:
    ov = load_overlay()
    nid = _next_id("u_", ov)
    ov["nodes"].append({"id": nid, "label": label or "node", "kind": kind or "custom", "desc": desc})
    save_overlay(ov)
    return {"ok": True, "id": nid}


def import_model(name: str, source: str, kind: str = "model", desc: str = "") -> dict:
    """Register a user model/LLM/repo as a 'hovering' node — unconnected until the user wires it in."""
    ov = load_overlay()
    nid = _next_id("m_", ov)
    ov["nodes"].append({"id": nid, "label": name or "model", "kind": kind, "source": source,
                        "desc": desc or f"Imported {kind}: {source}", "hovering": True})
    save_overlay(ov)
    return {"ok": True, "id": nid, "hovering": True}


def delete_node(node_id: str) -> dict:
    sp = _find_spine(node_id)
    if sp and sp.get("locked"):
        return _disclaimer({**sp})                       # locked: refuse + explain
    ov = load_overlay()
    if sp:                                                # optional system node -> disable, don't destroy
        if node_id not in ov["disabled"]:
            ov["disabled"].append(node_id)
        save_overlay(ov)
        return {"ok": True, "disabled": True, "id": node_id}
    if any(n["id"] == node_id for n in ov["nodes"]):      # user node -> remove
        if node_id not in ov["removed"]:
            ov["removed"].append(node_id)
        save_overlay(ov)
        return {"ok": True, "removed": True, "id": node_id}
    return {"ok": False, "error": "unknown node"}


def enable_node(node_id: str) -> dict:
    ov = load_overlay()
    ov["disabled"] = [d for d in ov["disabled"] if d != node_id]
    ov["removed"] = [r for r in ov["removed"] if r != node_id]
    save_overlay(ov)
    return {"ok": True, "id": node_id}


def connect(src: str, dst: str) -> dict:
    ov = load_overlay()
    if not any(e["from"] == src and e["to"] == dst for e in ov["edges"]):
        ov["edges"].append({"from": src, "to": dst})
    save_overlay(ov)
    return {"ok": True}


def disconnect(src: str, dst: str) -> dict:
    if (src, dst) in _LOCKED_EDGES:
        return {"ok": False, "locked": True,
                "disclaimer": "This connection is part of the locked core path; severing it would break "
                              "the flow from your data to the answer."}
    ov = load_overlay()
    ov["edges"] = [e for e in ov["edges"] if not (e["from"] == src and e["to"] == dst)]
    save_overlay(ov)
    return {"ok": True}


def set_position(node_id: str, x: float, y: float) -> dict:
    ov = load_overlay()
    ov["positions"][node_id] = [round(x), round(y)]
    save_overlay(ov)
    return {"ok": True}


def node_details(node_id: str) -> dict:
    wf = get_workflow()
    node = next((n for n in wf["nodes"] if n["id"] == node_id), None)
    if not node:
        return {"ok": False, "error": "unknown node"}
    incoming = [e["from"] for e in wf["edges"] if e["to"] == node_id]
    outgoing = [e["to"] for e in wf["edges"] if e["from"] == node_id]
    return {"ok": True, "id": node_id, "label": node["label"], "kind": node["kind"],
            "locked": node.get("locked", False), "reason": node.get("reason", ""),
            "description": node.get("desc", ""), "system": node.get("system", False),
            "optional": node.get("optional", False), "incoming": incoming, "outgoing": outgoing,
            "source": node.get("source", "")}


def list_graphs() -> list[dict]:
    """The menu of graphs/workflows + how each connects to the others (the side-panel content)."""
    return [
        {"id": "cognitive", "label": "Cognitive graph",
         "desc": "Your projects, files, and chunks with shared-field edges. The DATA layer.",
         "connects": "Feeds the Processing pipeline at the 'Your corpus' + 'Hybrid retrieval' nodes. "
                     "Read-only here; edit it with carto ingest / index."},
        {"id": "pipeline", "label": "Processing pipeline",
         "desc": "The editable LangGraph-style flow your prompt travels: capture → … → steering brief.",
         "connects": "Reads from the Cognitive graph and the live Vision stream; emits the Steering "
                     "brief consumed by your agents (Claude/Cursor/ChatGPT/Gemini)."},
        {"id": "vision", "label": "Vision stream",
         "desc": "Optional real-time screen capture that continuously grows the Cognitive graph.",
         "connects": "Writes redacted, classified activity INTO the Cognitive graph; surfaces as live "
                     "context inside the Steering brief."},
    ]


def validate() -> dict:
    """Sanity check the merged workflow: every locked node present, no edge to a dead node."""
    wf = get_workflow()
    ids = {n["id"] for n in wf["nodes"]}
    issues = []
    for n in _SPINE:
        if n.get("locked") and n["id"] not in ids:
            issues.append(f"locked node missing: {n['id']}")
    for e in wf["edges"]:
        if e["from"] not in ids or e["to"] not in ids:
            issues.append(f"dangling edge: {e['from']}->{e['to']}")
    return {"ok": not issues, "issues": issues, "nodes": len(ids), "edges": len(wf["edges"])}
