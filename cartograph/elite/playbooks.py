"""Frontier playbooks — the WORKFLOW (sequence of moves) a top practitioner follows, per field.
Not a checklist of artifacts (that's the DoD) — the process that distinguishes how the best work."""
from __future__ import annotations

from .catalog import match_field

PLAYBOOKS: dict[str, list[str]] = {
    "ml_experiment": [
        "Frame the problem + metric that matches the real objective; define a STRONG baseline first.",
        "Audit the data (leakage, label noise, distribution shift) before modeling.",
        "Iterate by ABLATION — change one thing, measure its isolated contribution.",
        "Do qualitative ERROR ANALYSIS on failure cases, not just aggregate metrics.",
        "Check calibration + report uncertainty; beat the baseline on HELD-OUT with CI.",
        "Pin seeds/config/data version; make every result reproducible.",
        "Report compute + wall-clock + cost; note negative results honestly.",
    ],
    "quant_research": [
        "Frame a falsifiable hypothesis with an economic rationale (not data-mined).",
        "Assemble point-in-time, survivorship-bias-free data; define universe + rebalance.",
        "Engineer features; guard against lookahead; document each transform.",
        "Validate with PURGED + EMBARGOED walk-forward CV (no leakage across the gap).",
        "Backtest with realistic costs, slippage, capacity, turnover.",
        "Stress across regimes; report deflated Sharpe + multiple-testing correction.",
        "Size by risk not signal; paper-trade before capital; pre-register the decision rule.",
    ],
    "hpc": [
        "PROFILE first (flame graph / Nsight) — never optimize on intuition.",
        "Place the kernel on the roofline; know if you're compute- or memory-bound.",
        "Improve algorithm/arithmetic intensity before micro-tuning.",
        "Tune memory access, occupancy, bandwidth; validate numerics/precision.",
        "Measure strong + weak scaling; report vs hardware peak.",
        "Re-profile to confirm the bottleneck moved; document speedup + method.",
    ],
    "agent_app": [
        "Define the EVAL HARNESS first (tasks + graders) — optimize against it, not vibes.",
        "Ship a baseline agent; instrument traces of prompts/tool-calls/cost.",
        "Design tool + prompt-injection guardrails; least-privilege tools.",
        "Iterate against the eval; regression-gate every change.",
        "Add graceful degradation when a model/API is down; budget latency + token cost.",
        "Red-team adversarially before claiming done.",
    ],
    "web_frontend": [
        "Start from the user journey + loading/empty/error/offline states, not the happy path.",
        "Build accessible-by-default (semantic HTML, keyboard, ARIA) — not a later pass.",
        "Set a performance budget; measure Core Web Vitals on real devices.",
        "Ship minimal JS; defer/stream; treat every kilobyte as a cost.",
        "Verify responsive + cross-browser; test the states, not just the render.",
    ],
    "data_pipeline": [
        "Model the data contract first (schema, freshness SLA, ownership).",
        "Make every step IDEMPOTENT and re-runnable; design backfills from day one.",
        "Assert data-quality at boundaries; quarantine bad rows, don't drop silently.",
        "Track lineage + freshness; make 'where did this number come from' answerable.",
        "Monitor cost + runtime + late data; alert on SLA breach, not just failures.",
    ],
    "research_paper": [
        "State the contribution + hypothesis crisply; position vs SOTA honestly.",
        "Design experiments that could FALSIFY the claim, not just confirm it.",
        "Build a reproducible artifact that regenerates every figure/table.",
        "Report significance/CIs; state limitations + threats to validity.",
    ],
    "library": [
        "Design the public API from the CALLER's code first; keep it minimal + composable.",
        "Type everything; type-check in CI; make illegal states unrepresentable.",
        "Write docstrings + runnable examples as you design — docs are an API tool.",
        "Cover edge cases + error paths; version with semver; deprecate before removing.",
    ],
    "devops": [
        "Declare infra as code; one source of truth; review changes like app code.",
        "Make deploys idempotent + reversible; automate rollback.",
        "Build observability in (logs/metrics/traces) before you need it.",
        "Least-privilege everything; secrets in a vault, never in code.",
    ],
}


def playbook_for(field_or_task: str) -> dict:
    f = match_field(field_or_task)
    if f and f in PLAYBOOKS:
        return {"field": f, "steps": PLAYBOOKS[f],
                "note": "the frontier process for this field — follow the sequence, keep the rigor"}
    return {"field": field_or_task or "general", "steps": [],
            "note": "no playbook yet for this field — add it to playbooks.py"}
