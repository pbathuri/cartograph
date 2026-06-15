"""`carto demo` — see the whole thing work in one command, with zero setup and zero data of your own.

Generates a tiny synthetic multi-field corpus in a temp workspace, ingests it, then walks through
retrieval, persona, personalization, and the elite layer — so a brand-new user (technical or not)
watches Cartograph build a graph and steer answers in ~10 seconds. Touches only a temp dir; your real
workspace is untouched."""
from __future__ import annotations

import tempfile
from pathlib import Path

# A few mini "projects" across different fields — representative enough to drive field inference,
# retrieval, and persona steering without cloning anything.
_PROJECTS = {
    "ml-lab": {
        "field": "ml_experiment",
        "files": {
            "train.py": "import logging\nlogger=logging.getLogger(__name__)\n"
                        "def train(X,y,seed=0):\n    '''Fit a baseline with a held-out split; report calibrated metrics.'''\n"
                        "    from sklearn.linear_model import LogisticRegression\n"
                        "    from sklearn.model_selection import train_test_split\n"
                        "    Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=seed)\n"
                        "    try:\n        m=LogisticRegression().fit(Xtr,ytr)\n    except Exception as e:\n        logger.error('%s',e); raise\n"
                        "    return {'acc':m.score(Xte,yte)}\n",
            "README.md": "# ml-lab\nTransformer + baseline experiments. Ablations, calibration, held-out metrics.",
        },
    },
    "web-store": {
        "field": "web_frontend",
        "files": {
            "package.json": '{"name":"web-store","dependencies":{"react":"^18","next":"14"}}',
            "Cart.tsx": "export function Cart(){\n  // loading/empty/error states, aria-labels\n  return <button aria-label='checkout'/>\n}",
            "README.md": "# web-store\nNext.js storefront. Core Web Vitals budgeted; accessible by default.",
        },
    },
    "etl-warehouse": {
        "field": "data_pipeline",
        "files": {
            "dbt_project.yml": "name: warehouse\nversion: 1",
            "staging.sql": "-- idempotent, backfill-safe\nselect * from raw.events where loaded_at >= '{{ var(\"start\") }}'",
            "README.md": "# etl-warehouse\ndbt models with data-quality tests + lineage.",
        },
    },
    "typed-sdk": {
        "field": "library",
        "files": {
            "pyproject.toml": "[project]\nname='typed-sdk'\nversion='0.1.0'",
            "client.py": "__all__=['Client']\nclass Client:\n    '''Minimal typed public API.'''\n    def get(self, url: str) -> dict:\n        ...\n",
            "README.md": "# typed-sdk\nA small, fully-typed Python library. pip install typed-sdk.",
        },
    },
}


def write_corpus(root: Path) -> Path:
    for name, spec in _PROJECTS.items():
        for fn, body in spec["files"].items():
            p = root / name / fn
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
    return root


def run_demo(console) -> None:
    import os

    from .config import Config
    from .elite import elevate, frontier_report
    from .persona import build_brief, record_feedback
    from .persona.profile import build_from_corpus, save_persona
    from .retrieve import retrieve
    from .storage import Store

    tmp = Path(tempfile.mkdtemp(prefix="cartograph_demo_"))
    corpus = write_corpus(tmp / "corpus")
    os.environ["CARTOGRAPH_HOME"] = str(tmp / "home")
    cfg = Config(field_focus=[])
    store = Store(tmp / "home" / "graph.sqlite")

    console.print("\n[bold]1. Ingest[/bold] — building a graph from a synthetic multi-field corpus…")
    from .ingest import ingest_path
    st = ingest_path(corpus, store, cfg)
    console.print(f"   {st.projects} projects · {st.files_indexed} files · {st.chunks} chunks · "
                  f"fields: {', '.join(st.fields)}")

    console.print("\n[bold]2. Retrieve[/bold] — keyword search across everything (semantic auto-on if installed):")
    r = retrieve("validate input data and handle errors", store, cfg, top_k=3)
    for c in r.chunks[:3]:
        console.print(f"   [cyan]{c.get('project_name')}[/cyan]  {(c.get('chunk_text') or '')[:60].strip()}")

    console.print("\n[bold]3. Persona[/bold] — learned from the corpus, then a feedback signal:")
    p = build_from_corpus(store)
    save_persona(p)
    p = record_feedback(p, store, cfg, query="typed api", liked_projects=["typed-sdk"], weight=2.0)
    console.print(f"   {p.summary()}")

    console.print("\n[bold]4. Personalize[/bold] — the steering brief an agent prepends:")
    b = build_brief("how should I design my API?", store, cfg, p)
    console.print(f"   prompt field: [magenta]{b['prompt_field']}[/magenta]")
    for g in b["output_guidance"][:3]:
        console.print(f"   - {g}")

    console.print("\n[bold]5. Elevate[/bold] — top-of-field briefing for a task:")
    e = elevate("build a clean typed python library", store, cfg)
    console.print(f"   field: {e['field']}  | refs: {', '.join(e['references'][:3])}")

    console.print("\n[bold]6. Frontier[/bold] — coverage of each field's best references:")
    fr = frontier_report(store, top=3)
    for f in fr["fields"][:4]:
        console.print(f"   {f['field']:14s} {f['coverage_pct']}%")

    console.print("\n[green]✓ That's Cartograph.[/green] Now point it at YOUR work:")
    console.print("   [bold]carto init && carto ingest ~/your-folders && carto viz[/bold]")
    console.print(f"[dim]   (demo workspace was {tmp} — safe to delete)[/dim]")
