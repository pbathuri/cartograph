"""Elite Definition-of-Done — the bar a top practitioner holds, per field. A universal professional
bar + field-specific elite practices. `score_build` heuristically grades a build by scanning file
names + content markers. Honest by construction: a marker shows a practice was REFERENCED, not that
it is correct — confirm with eval/human review."""
from __future__ import annotations

import re
from pathlib import Path

from .catalog import match_field

UNIVERSAL = [
    ("tests", "automated tests cover the core paths and pass"),
    ("docs", "README + how-to-run + decisions captured"),
    ("error_handling", "failure modes handled; states: loading/empty/error"),
    ("reproducibility", "deterministic build/run; pinned deps; seeds where stochastic"),
    ("security", "no secrets committed; inputs validated; least-privilege"),
    ("observability", "logging/metrics for the critical path"),
]

ELITE = {
    "ml_experiment": [("ablations", "ablation isolating each component"), ("calibration", "calibrated confidence"),
                      ("baselines", "strong baseline beaten on held-out with CI"), ("error_analysis", "qualitative failure analysis")],
    "quant_research": [("walk_forward", "purged/embargoed walk-forward"), ("transaction_costs", "realistic costs+slippage"),
                       ("significance", "deflated Sharpe / multiple-testing"), ("data_hygiene", "point-in-time, no lookahead")],
    "hpc": [("profiling", "profiled before optimizing"), ("roofline", "arithmetic-intensity vs peak"),
            ("scaling", "strong+weak scaling measured")],
    "agent_app": [("eval_harness", "automated eval, regression-gated"), ("safety", "prompt-injection + tool guardrails"),
                  ("observability", "traces of prompts/tools/cost")],
    "web_frontend": [("a11y", "WCAG + keyboard/screen-reader"), ("perf", "Core Web Vitals budgeted"),
                     ("states", "loading/empty/error/offline")],
    "data_pipeline": [("idempotent", "re-runnable, backfill-safe"), ("data_quality", "schema + quality checks"),
                      ("lineage", "lineage + freshness tracked")],
    "research_paper": [("repro_artifact", "code+data reproduce every figure"), ("related_work", "positioned vs SOTA"),
                       ("limitations", "limitations + threats stated")],
    "library": [("typed_api", "fully typed, minimal public API"), ("docs_examples", "docstrings + runnable examples"),
                ("coverage", "edge cases covered")],
    "devops": [("iac", "infrastructure as code"), ("reversible", "automated rollback"), ("least_privilege", "scoped secrets")],
}

_FILE_MARKERS = {
    "tests": re.compile(r"(^|/)tests?/|test_|_test\.|\.test\.|\.spec\.", re.I),
    "docs": re.compile(r"readme", re.I),
    "security": re.compile(r"\.gitignore|\.env\.example", re.I),
    "reproducibility": re.compile(r"requirements|lock|pyproject|environment\.ya?ml", re.I),
}
_CONTENT_MARKERS = {
    "error_handling": re.compile(r"try:|except\b|raise\b|catch\(|finally", re.I),
    "observability": re.compile(r"logging|logger|getlogger|metric|trace|telemetry", re.I),
    "ablations": re.compile(r"ablation|leave.?one.?out", re.I),
    "calibration": re.compile(r"calibrat|reliability.?diagram|brier|isotonic", re.I),
    "baselines": re.compile(r"baseline|held.?out|cross.?val|confidence.?interval", re.I),
    "error_analysis": re.compile(r"error.?analysis|confusion.?matrix|failure.?case", re.I),
    "walk_forward": re.compile(r"walk.?forward|purged|embargo|timeseriessplit", re.I),
    "transaction_costs": re.compile(r"slippage|commission|transaction.?cost|turnover", re.I),
    "significance": re.compile(r"deflated.?sharpe|p.?value|bootstrap|bonferroni", re.I),
    "data_hygiene": re.compile(r"point.?in.?time|survivorship|lookahead|as.?of", re.I),
    "profiling": re.compile(r"profil|nsight|flame.?graph|cprofile|nvprof", re.I),
    "roofline": re.compile(r"roofline|arithmetic.?intensity|flop|bandwidth", re.I),
    "scaling": re.compile(r"strong.?scal|weak.?scal|speedup|amdahl", re.I),
    "eval_harness": re.compile(r"eval|promptfoo|ragas|grader|regression.?test", re.I),
    "safety": re.compile(r"prompt.?inject|guardrail|sanitiz|least.?privilege", re.I),
    "a11y": re.compile(r"aria-|role=|wcag|a11y|alt=", re.I),
    "perf": re.compile(r"lighthouse|core.?web.?vital|lcp|cls|inp", re.I),
    "states": re.compile(r"loading|empty.?state|error.?state|skeleton|offline", re.I),
    "idempotent": re.compile(r"idempoten|upsert|backfill|merge.?into", re.I),
    "data_quality": re.compile(r"great_expectation|schema|validation|dbt.?test", re.I),
    "lineage": re.compile(r"lineage|provenance|freshness|airflow|dagster", re.I),
    "repro_artifact": re.compile(r"reproduc|makefile|figure|notebook", re.I),
    "related_work": re.compile(r"related.?work|sota|state.?of.?the.?art", re.I),
    "limitations": re.compile(r"limitation|threat.?to.?valid|future.?work", re.I),
    "typed_api": re.compile(r"__all__|->\s*\w|: \w+ =|py\.typed", re.I),
    "docs_examples": re.compile(r">>>|examples?/|usage|\"\"\"", re.I),
    "coverage": re.compile(r"coverage|pytest|parametrize|edge.?case", re.I),
    "iac": re.compile(r"terraform|\.tf\b|cloudformation|pulumi", re.I),
    "reversible": re.compile(r"rollback|blue.?green|canary", re.I),
    "least_privilege": re.compile(r"least.?privilege|iam|secret|vault", re.I),
}
_TEXT_EXT = {".py", ".md", ".js", ".ts", ".tsx", ".rst", ".txt", ".yaml", ".yml", ".toml",
             ".html", ".css", ".tf", ".sql", ".ipynb", ".java", ".go", ".rs", ".cu", ".cpp"}


def dod_for(field_or_task: str) -> dict:
    f = match_field(field_or_task) or "general"
    universal = [{"dim": d, "req": r, "tier": "universal"} for d, r in UNIVERSAL]
    elite = [{"dim": d, "req": r, "tier": "elite"} for d, r in ELITE.get(f, [])]
    return {"field": f, "criteria": universal + elite}


def score_build(project: str | Path, field_or_task: str) -> dict:
    root = Path(project)
    crit = dod_for(field_or_task)
    names, content = [], ""
    if root.is_dir():
        files = [p for p in root.rglob("*") if p.is_file()][:5000]
        names = [str(p).lower() for p in files]
        nread = 0
        for p in files:
            if p.suffix.lower() in _TEXT_EXT and nread < 300:
                try:
                    content += p.read_text(encoding="utf-8", errors="ignore")[:40000]
                    nread += 1
                except Exception:
                    pass
    blob = " ".join(names)
    met, unmet = [], []
    for c in crit["criteria"]:
        dim = c["dim"]
        hit = False
        fm = _FILE_MARKERS.get(dim)
        if fm and fm.search(blob):
            hit = True
        if not hit:
            cm = _CONTENT_MARKERS.get(dim)
            hit = bool(cm and content and cm.search(content))
        (met if hit else unmet).append(f"[{c['tier']}] {dim}: {c['req']}")
    elite_total = sum(1 for c in crit["criteria"] if c["tier"] == "elite")
    elite_met = sum(1 for m in met if m.startswith("[elite]"))
    univ_met = sum(1 for m in met if m.startswith("[universal]"))
    if elite_total and elite_met >= max(1, elite_total // 2) and univ_met >= 4:
        grade = "elite"
    elif univ_met >= 4:
        grade = "professional"
    elif univ_met >= 2:
        grade = "baseline"
    else:
        grade = "minimal"
    return {"field": crit["field"], "grade": grade, "met": met, "unmet": unmet,
            "note": "heuristic: filename + content-marker scan. A marker shows a practice was referenced, "
                    "NOT that it is correct — confirm with eval/human review."}
