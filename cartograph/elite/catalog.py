"""Curated frontier reference repos per field — the canonical public OSS that leading practitioners
use, mapped to the elite practice each teaches. Field-agnostic and extensible: add your field here.
All entries are public, license-clean OSS. Nothing here is personal to any user."""
from __future__ import annotations

import re

# field -> list of (repo, teaches_practice, why_top_tier)
CATALOG: dict[str, list[tuple[str, str, str]]] = {
    "ml_experiment": [
        ("huggingface/transformers", "baselines", "reference implementations + strong baselines"),
        ("EleutherAI/lm-evaluation-harness", "eval_harness", "standardized LLM eval (the field's bar)"),
        ("scikit-learn/scikit-learn", "calibration", "calibration, model selection, metrics done right"),
        ("karpathy/nanoGPT", "repro", "minimal, reproducible training reference"),
        ("wandb/wandb", "experiment_tracking", "experiment tracking, sweeps, reproducibility"),
    ],
    "quant_research": [
        ("microsoft/qlib", "walk_forward", "production AI-quant platform; rolling retrain, point-in-time"),
        ("hudson-and-thames/mlfinlab", "data_hygiene", "AFML: triple-barrier, deflated Sharpe, purged CV"),
        ("robcarver17/pysystemtrade", "risk", "systematic futures: sizing, costs, capacity"),
        ("polakowo/vectorbt", "backtesting", "vectorized backtesting with realistic fills"),
        ("ranaroussi/quantstats", "risk_reporting", "tearsheets: drawdown, tail risk, regimes"),
    ],
    "hpc": [
        ("triton-lang/triton", "profiling", "GPU kernel authoring; occupancy/bandwidth-aware"),
        ("NVIDIA/cutlass", "roofline", "high-performance GEMM; arithmetic-intensity tuning"),
        ("rapidsai/cudf", "scaling", "GPU dataframes; scale-out data"),
        ("ggml-org/llama.cpp", "quantization", "extreme quantization + CPU/GPU efficiency"),
    ],
    "agent_app": [
        ("promptfoo/promptfoo", "eval_harness", "agent/prompt eval + regression gating"),
        ("explodinggradients/ragas", "rag_eval", "RAG eval (context precision/recall)"),
        ("langfuse/langfuse", "observability", "LLM tracing/cost observability"),
        ("microsoft/autogen", "orchestration", "multi-agent orchestration + guardrails"),
    ],
    "web_frontend": [
        ("vercel/next.js", "perf", "RSC, Core Web Vitals best practices"),
        ("withastro/astro", "perf", "islands architecture; minimal JS"),
        ("radix-ui/primitives", "a11y", "accessible component primitives"),
    ],
    "data_pipeline": [
        ("apache/airflow", "orchestration", "DAG orchestration; retries, backfills, idempotency"),
        ("dbt-labs/dbt-core", "data_quality", "tested, documented, lineage-tracked transforms"),
        ("great-expectations/great_expectations", "data_quality", "declarative data-quality suites"),
        ("dagster-io/dagster", "lineage", "asset-based orchestration; data-aware scheduling"),
    ],
    "research_paper": [
        ("papers-we-love/papers-we-love", "related_work", "curated foundational CS papers"),
        ("iterative/dvc", "repro_artifact", "data+experiment versioning; reproducible results"),
        ("executablebooks/jupyter-book", "repro_artifact", "executable, reproducible publications"),
    ],
    "library": [
        ("pydantic/pydantic", "typed_api", "typed models; ergonomic, validated public API"),
        ("psf/requests", "api_design", "API ergonomics + docs as a design discipline"),
        ("tiangolo/typer", "docs_examples", "docstring-driven CLI with runnable examples"),
    ],
    "devops": [
        ("kubernetes/kubernetes", "orchestration", "container orchestration reference"),
        ("hashicorp/terraform", "iac", "declarative infrastructure-as-code"),
        ("ansible/ansible", "config_mgmt", "idempotent configuration management"),
    ],
    "mobile": [
        ("flutter/flutter", "cross_platform", "single-codebase native UI"),
        ("pointfreeco/swift-composable-architecture", "state_mgmt", "testable app state architecture"),
    ],
    "game_dev": [
        ("godotengine/godot", "engine", "open-source engine; scene/node architecture"),
        ("raysan5/raylib", "fundamentals", "simple, dependency-light game primitives"),
    ],
}

ADJOINING: dict[str, list[str]] = {
    "quant_research": ["ml_experiment (eval rigor, calibration)", "hpc (backtest speed)"],
    "ml_experiment": ["hpc (training efficiency)", "agent_app (eval-harness discipline)"],
    "agent_app": ["ml_experiment (eval rigor)", "web_frontend (UX states)"],
    "data_pipeline": ["ml_experiment (data hygiene)", "hpc (throughput)"],
}

_KW = {
    "quant_research": ("quant", "trading", "backtest", "factor", "portfolio", "alpha", "finance"),
    "ml_experiment": ("ml", "model", "train", "deep", "learning", "neural", "llm"),
    "agent_app": ("agent", "llm", "chat", "rag", "prompt", "assistant"),
    "hpc": ("hpc", "cuda", "gpu", "kernel", "parallel"),
    "web_frontend": ("web", "frontend", "ui", "react", "css"),
    "data_pipeline": ("pipeline", "etl", "ingest", "warehouse", "dbt", "airflow"),
    "library": ("library", "package", "sdk", "framework", "api"),
    "research_paper": ("paper", "research", "thesis", "arxiv"),
    "devops": ("devops", "kubernetes", "docker", "terraform", "infra", "deploy"),
    "mobile": ("mobile", "android", "ios", "flutter", "swift"),
    "game_dev": ("game", "unity", "godot", "shader", "sprite"),
}


def _kw_hit(text: str, kw: str) -> bool:
    return re.search(rf"\b{re.escape(kw)}\b", text) is not None


def match_field(key: str) -> str | None:
    k = (key or "").lower().strip()
    for f in CATALOG:
        if k == f or k in f or f in k:
            return f
    for f, kws in _KW.items():
        if any(_kw_hit(k, w) for w in kws):
            return f
    return None


def elite_refs(field_or_task: str) -> dict:
    f = match_field(field_or_task)
    if f is None:
        return {"field": field_or_task, "references": [], "adjoining": [],
                "note": "no field match — add it to catalog.py or broaden the query"}
    return {"field": f,
            "references": [{"repo": r, "teaches": t, "why": w} for r, t, w in CATALOG[f]],
            "adjoining": ADJOINING.get(f, []),
            "note": "curated public OSS; clone + ingest with `carto ingest` to fold into your graph"}
