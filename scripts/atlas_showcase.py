"""Build a synthetic showcase graph (no personal data) for screenshots and demos of the Atlas.

    python scripts/atlas_showcase.py /tmp/carto-showcase      # then:
    CARTOGRAPH_HOME=/tmp/carto-showcase/home carto viz
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

FIELDS = {
    "research_paper": (["ocular-pk", "corneal-transport", "tear-film-model", "retina-imaging", "protein-folding",
                        "drug-release-hydrogel", "lab-notebook-2026", "thesis-chapters"],
                       ["numpy", "scipy", "pandas", "matplotlib", "lmfit", "sympy", "seaborn"]),
    "ml_experiment": (["vit-finetune", "segmentation-unet", "contrastive-pretrain", "llm-eval-harness",
                       "rag-baselines", "diffusion-sampler", "tabular-boosting", "speech-asr"],
                      ["torch", "transformers", "datasets", "accelerate", "scikit-learn", "wandb", "peft", "timm"]),
    "web_frontend": (["lab-website", "dashboard-react", "docs-portal", "annotation-tool"],
                     None),
    "quant_research": (["factor-zoo", "options-vol-surface", "pairs-backtest", "risk-parity"],
                       ["numpy", "pandas", "statsmodels", "cvxpy", "arch", "yfinance"]),
    "hpc": (["mpi-stencil", "cuda-nbody", "slurm-pipelines", "benchmark-suite"],
            ["numpy", "mpi4py", "numba", "cupy", "h5py"]),
    "agent_app": (["mcp-tools", "agent-orchestrator", "browser-agent", "eval-dashboard", "memory-service"],
                  ["httpx", "pydantic", "fastapi", "openai", "anthropic", "tenacity", "rich"]),
}
NPM = ["react", "next", "tailwindcss", "zustand", "d3", "vite", "typescript", "@tanstack/react-query"]
PAPERS = ["2401.04444", "1706.03762", "2005.11401", "2310.06825", "2203.02155", "1810.04805", "2106.09685",
          "2312.10997", "2404.16130", "2210.03629"]
MODELS = ["meta-llama/Llama-3.1-8B", "google/vit-base-patch16-224", "openai/whisper-small",
          "nomic-ai/nomic-embed-text-v1.5", "facebook/esm2_t33_650M_UR50D"]
DATASETS = ["imagenet-1k", "squad", "mteb/scifact", "glue", "librispeech_asr"]
REPOS = ["huggingface/transformers", "pytorch/pytorch", "langchain-ai/langgraph", "microsoft/graphrag",
         "facebookresearch/faiss", "modelcontextprotocol/servers", "vercel/next.js", "pydantic/pydantic"]
NOTES = {
    "research_paper": [
        "Transcorneal flux of timolol in Franz diffusion cells was {v} µg/cm²/h (n={n}, 34 °C).",
        "Apparent permeability Papp rose {p}% after removing the epithelium; stroma is not rate-limiting.",
        "Tear turnover of 16%/min explains most of the precorneal loss in the first 5 minutes.",
        "Hydrogel release follows a burst phase then Higuchi kinetics; fit R² 0.{p}.",
        "Reviewer 2 asked for the partition coefficient at pH 7.4, rerun with {n} replicates.",
        "Figure 3 compares the two-compartment PK model with aqueous humor samples from {n} rabbits.",
    ],
    "ml_experiment": [
        "Fine-tuning with LoRA rank 16 reached {p}% top-1 after {n} epochs; full fine-tune only +0.4.",
        "Eval harness: nDCG@10 {v} on SciFact with the hybrid retriever, BM25 alone lower.",
        "Mixed precision bf16 halves memory; gradient checkpointing needed above batch {n}.",
        "Data leak found: {n} near-duplicate images between train and test, deduplicated with pHash.",
        "Warmup of {n}00 steps stabilises loss spikes seen at the start of contrastive pretraining.",
    ],
    "web_frontend": [
        "Dashboard renders {n}k rows with virtualisation; React Query caches the experiment list.",
        "Lighthouse performance {p} after code-splitting the chart bundle.",
        "Annotation tool exports COCO JSON; keyboard shortcuts cut labelling time by {p}%.",
    ],
    "quant_research": [
        "Momentum factor Sharpe {v} gross, falls by a third after 5 bp costs and next-day fills.",
        "Vol surface fit uses SVI per expiry; arbitrage checks flag {n} butterflies.",
        "Pairs backtest: half-life of the spread {n} days, cointegration p-value 0.0{n}.",
    ],
    "hpc": [
        "Stencil kernel reaches {p}% of peak memory bandwidth on A100 with shared-memory tiling.",
        "MPI halo exchange dominates above {n} nodes; overlapping comm and compute saves {p}% wall time.",
        "Slurm array jobs with {n} tasks each; checkpoint every 30 minutes for preemptible queues.",
    ],
    "agent_app": [
        "MCP tool schemas validated against the spec; read-only tools annotated with readOnlyHint.",
        "Agent retries with exponential backoff; {n} max attempts, jitter avoids thundering herd.",
        "Memory service p95 latency {n} ms on SQLite WAL with {n} concurrent writers.",
        "Browser agent fails on {p}% of captcha pages; route those to a human.",
    ],
}
MEMORIES = [
    ("corneal-transport", "decision", "Use Franz diffusion cells at 34 °C for all transcorneal flux runs"),
    ("ocular-pk", "rejected", "Ussing chamber for timolol: too much edge damage, numbers not reproducible"),
    ("rag-baselines", "decision", "Evaluate retrieval with nDCG@10 on SciFact before any reranker change"),
    ("thesis-chapters", "preference", "Plots in seaborn whitegrid, 300 dpi, colour-blind palette"),
    ("drug-release-hydrogel", "todo", "Re-run the hydrogel release fits with the corrected burst-phase model"),
    ("ocular-pk", "fact", "Rabbit corneal thickness used in the PK model is 0.40 mm"),
    ("memory-service", "rejected", "Pinecone for the memory service: we need it to run offline"),
    ("memory-service", "decision", "FastAPI + SQLite for the memory service; Postgres only if we pass 10 writers"),
    ("llm-eval-harness", "session", "Fixed flaky tokenizer test, pinned transformers, re-ran eval harness"),
    ("mpi-stencil", "todo", "Port the stencil kernel to CUDA graphs and compare against the MPI baseline"),
    ("pairs-backtest", "decision", "Backtests use next-day open fills and 5 bp cost"),
    ("thesis-chapters", "session", "Drafted chapter 3 methods, added tear-film turnover citations"),
]


def build(root: Path, seed: int = 11) -> Path:
    rng = random.Random(seed)
    lab = root / "lab"
    for field, (names, pkgs) in FIELDS.items():
        for name in names:
            d = lab / name
            (d / ".git").mkdir(parents=True, exist_ok=True)
            papers = rng.sample(PAPERS, rng.randint(1, 3))
            text = [f"# {name}\n", f"Project in {field.replace('_', ' ')}.\n"]
            text += [f"Builds on arXiv:{p}." for p in papers]
            if field in ("ml_experiment", "agent_app", "research_paper") and rng.random() < .7:
                m = rng.choice(MODELS)
                text.append(f"Model: https://huggingface.co/{m}")
            if field == "ml_experiment" and rng.random() < .7:
                text.append(f"Data: https://huggingface.co/datasets/{rng.choice(DATASETS)}")
            text += [f"See https://github.com/{r}" for r in rng.sample(REPOS, rng.randint(0, 2))]
            (d / "README.md").write_text("\n".join(text) + "\n")
            if pkgs:
                deps = rng.sample(pkgs, rng.randint(3, min(6, len(pkgs))))
                (d / "requirements.txt").write_text("\n".join(deps) + "\n")
                (d / "main.py").write_text("".join(f"import {p.replace('-', '_')}\n" for p in deps[:3]) +
                                           "\n\ndef run():\n    return 'ok'\n")
            else:
                deps = rng.sample(NPM, 5)
                (d / "package.json").write_text('{\n  "name": "%s",\n  "dependencies": {\n%s\n  }\n}\n' % (
                    name, ",\n".join(f'    "{p}": "^1.0.0"' for p in deps)))
            lines = rng.sample(NOTES[field], min(len(NOTES[field]), 5))
            (d / "notes.md").write_text(f"## {name}: working notes\n\n" + "\n\n".join(
                s.format(n=rng.randint(3, 12), v=round(rng.uniform(0.5, 9.5), 2), p=rng.randint(60, 99))
                for s in lines) + "\n")
    return lab


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/carto-showcase").resolve()
    os.environ["CARTOGRAPH_HOME"] = str(root / "home")
    from cartograph import memory as M
    from cartograph.config import Config, db_path
    from cartograph.ingest import ingest_path
    from cartograph.storage import Store
    lab = build(root)
    store = Store(db_path())
    for d in sorted(lab.iterdir()):
        ingest_path(d, store, Config())
    rng = random.Random(3)
    now = time.time()
    for i, (proj, kind, text) in enumerate(MEMORIES):
        r = M.remember(store, text, kind, cwd=str(lab / proj), anchors=[str(lab / proj / "notes.md")])
        when = now - (len(MEMORIES) - i) * 86400 * rng.uniform(1.5, 3.5)
        with store.cursor() as c:
            c.execute("UPDATE memories SET created_at = ?, valid_from = ? WHERE id = ?", (when, when, r["id"]))
    old = M.recall(store, "FastAPI SQLite memory service")["memories"][0]
    new = M.remember(store, "Memory service: FastAPI + SQLite in WAL mode; move to Postgres past 10 concurrent writers",
                     "decision", cwd=str(lab / "memory-service"))
    M.supersede(store, old["id"], new["id"], reason="load test showed WAL handles our write rate")
    print(f"showcase graph: {store.stats()}\nCARTOGRAPH_HOME={root / 'home'} carto viz")


if __name__ == "__main__":
    main()
