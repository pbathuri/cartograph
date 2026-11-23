"""`carto demo` — see the whole thing work in one command, with zero setup and zero data of your own.

Writes a tiny synthetic lab (a thesis in Word, slides and PDF notes, model code, a notebook, a dashboard)
to a temp workspace, then shows what an agent gets: cited answers, a graph built without an LLM, memory
that survives sessions and notices when files change, and the clients it plugs into. Your real
workspace is untouched. `write_corpus` is a separate multi-field corpus used by the tests."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
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


_UNET = "arXiv:1505.04597"
_TRAIN = ("import segmentation_models_pytorch as smp\nimport torch\n\n\n"
          "def loss_fn(pred, target):\n    \"\"\"Dice + BCE: BCE alone misses thin vessels.\"\"\"\n"
          "    return smp.losses.DiceLoss('binary')(pred, target) + torch.nn.BCEWithLogitsLoss()(pred, target)\n\n\n"
          "def train(model, loader, epochs=40, lr=3e-4):\n    opt = torch.optim.AdamW(model.parameters(), lr=lr)\n"
          "    for _ in range(epochs):\n        for x, y in loader:\n            opt.zero_grad()\n"
          "            loss_fn(model(x), y).backward()\n            opt.step()\n")


def write_lab(root: Path) -> Path:
    """A small lab whose projects share a paper, a model and a repo, in Word, PowerPoint, PDF, a notebook and code."""
    from .bench import _docx_bytes, _pptx_bytes, pdf_bytes
    cells = [("markdown", "# Held-out evaluation"),
             ("code", "scores = evaluate(model, heldout)  # 96 images from site C"),
             ("markdown", "Held-out Dice 0.87 with Dice + BCE; BCE alone gave 0.81. Thin vessels improved most.")]
    nb = {"cells": [{"cell_type": t, "source": s, "metadata": {}, **({"outputs": [], "execution_count": None}
                                                                    if t == "code" else {})} for t, s in cells],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    files: dict[str, str | bytes] = {
        "thesis/chapter3.docx": _docx_bytes([
            ("Methods", f"We train a U-Net ({_UNET}) on 412 fundus images from three sites, using a Dice plus "
                        "BCE loss and the weights published at huggingface.co/retina-lab/unet-vessels."),
            ("Results", "On the held-out set from site C the model reaches a Dice score of 0.87 for retinal "
                        "vessels, up from 0.81 with BCE alone.")]),
        "thesis/defense.pptx": _pptx_bytes(["Retinal vessel segmentation", "Data: 412 fundus images, 3 sites",
                                            "Held-out Dice 0.87 (BCE alone: 0.81)", "Next: thin-vessel recall"]),
        "thesis/related-work.pdf": pdf_bytes([
            f"Related work. U-Net ({_UNET}) remains the standard baseline for retinal vessel segmentation.",
            "Thin vessels under 3 px are the main failure mode. Overlap-based losses such as Dice help most."]),
        "retina-seg/requirements.txt": "torch\nsegmentation-models-pytorch\nalbumentations\n",
        "retina-seg/train.py": _TRAIN,
        "retina-seg/README.md": (f"# retina-seg\nU-Net vessel segmentation ({_UNET}).\n"
                                 "Weights: https://huggingface.co/retina-lab/unet-vessels\n"
                                 "Based on https://github.com/qubvel/segmentation_models.pytorch\n"),
        "lab-notebook/analysis.ipynb": json.dumps(nb),
        "lab-notebook/README.md": ("# lab-notebook\nEvaluation runs for the thesis. Model: "
                                   "https://huggingface.co/retina-lab/unet-vessels\n"),
        "dashboard/package.json": '{"name": "dashboard", "dependencies": {"react": "^18", "vite": "^5"}}',
        "dashboard/App.tsx": "export function App() {\n  return <main aria-label='results'>Dice by site</main>\n}\n",
    }
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body) if isinstance(body, bytes) else p.write_text(body, encoding="utf-8")
        (p.parent / ".git").mkdir(exist_ok=True)
    return root


def run_demo(console) -> None:
    import os

    from . import connect as C
    from . import entities as E
    from . import memory as M
    from .config import Config
    from .documents import pdf_backend
    from .ingest import ingest_path
    from .retrieve import context
    from .storage import Store

    tmp = Path(tempfile.mkdtemp(prefix="cartograph_demo_"))
    lab = write_lab(tmp / "lab")
    old_home = os.environ.get("CARTOGRAPH_HOME")
    os.environ["CARTOGRAPH_HOME"] = str(tmp / "home")
    cfg = Config(field_focus=[])
    store = Store(tmp / "home" / "graph.sqlite")
    prefixes = (str(lab.resolve()) + os.sep, str(lab) + os.sep)
    short = lambda cite: next((cite[len(p):] for p in prefixes if cite.startswith(p)), cite)  # noqa: E731
    width = max(24, min(console.width, 110) - 40)

    def preview(text: str, cite: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1 and lines[0].lstrip("# ") in cite:      # the heading is already in the citation
            lines = lines[1:]
        s = " ".join(" ".join(lines).split())
        return s if len(s) <= width else s[:width - 1] + "…"

    try:
        t = time.perf_counter()
        st = ingest_path(lab, store, cfg)
        console.print(f"\n[bold]1. Ingest[/bold]  {st.projects} projects · {st.files_indexed} files · {st.chunks} chunks "
                      f"in {time.perf_counter() - t:.1f} s  [dim]Word, PowerPoint, PDF, notebook, code[/dim]")
        if not pdf_backend():
            console.print("   [dim]PDFs are skipped until you install the docs extra: "
                          "pip install 'cartograph__v1\\[docs]'[/dim]")

        q = "what Dice score did we get on the held-out set?"
        console.print(f'\n[bold]2. Ask[/bold]  "{q}"  [dim]answers cite their source[/dim]')
        for ch in context(q, store, cfg, mode="chunks", max_tokens=600, top_k=4)["chunks"][:4]:
            console.print(f"   [cyan]{short(ch['cite']):<34}[/cyan] {preview(ch['text'], ch['cite'])}")

        console.print("\n[bold]3. Graph[/bold]  built from the files themselves, no LLM:")
        bare = lambda e: e.split(":", 1)[1]  # noqa: E731   "paper:arxiv:1505.04597" -> "arxiv:1505.04597"
        for f in E.find(store, _UNET.split(":")[1])[:3]:
            ev = short(f.get("evidence", {}).get("cite", ""))
            console.print(f"   [bold]{f['project']:<11}[/bold] cites [magenta]{bare(f['entity'])}[/magenta]  [dim]← {ev}[/dim]")
        pid = M.project_for(store, cwd=str(lab / "retina-seg"))
        for r in E.related_projects(store, pid)[:2] if pid else []:
            console.print(f"   [bold]retina-seg[/bold] [gold1]⇄[/gold1] [bold]{r['project']}[/bold]  "
                          f"[dim]linked: both use {', '.join(bare(v) for v in r.get('via', [])[:3])}[/dim]")

        seg = str(lab / "retina-seg")
        d = M.remember(store, "Use Dice + BCE loss: BCE alone missed thin vessels", "decision",
                       cwd=seg, anchors=[str(lab / "retina-seg" / "train.py")], source="demo")
        M.remember(store, "Random crops: they cut vessels at the image border", "rejected", cwd=seg, source="demo")
        console.print("\n[bold]4. Memory[/bold]  saved by one agent; the next session, in any client, starts with:")
        for line in M.brief(store, cwd=seg)["text"].splitlines():
            line = line.strip()
            if line.startswith("## ") or (line.startswith("**") and line.endswith("**")):
                console.print(f"   [dim]│[/dim] [bold]{line.strip('#* ')}[/bold]")
            elif line:
                console.print(f"   [dim]│[/dim] {line}")

        (lab / "retina-seg" / "train.py").write_text(_TRAIN.replace("DiceLoss('binary')(pred, target) + ", ""),
                                                     encoding="utf-8")
        ingest_path(lab, store, cfg)
        if (M.get(store, d["id"]) or {}).get("stale"):
            console.print(f"   [yellow]train.py just changed, so decision [{d['id']}] is now flagged as possibly stale[/yellow]")

        found = [C.CLIENTS[c][0] for c in C.CLIENTS if C.detected(c)]
        console.print("\n[bold]5. Plug in[/bold]  [bold]carto connect all[/bold] adds it to "
                      + (f"the clients found here: {', '.join(found)}" if found else
                         "Claude Code, Cursor, Codex, Gemini CLI, VS Code, Zed and 6 more"))

        console.print("\n[green]✓ That's Cartograph.[/green] Now on your own files:")
        console.print("   [bold]pip install 'cartograph__v1\\[docs]'[/bold]   # PDF support")
        console.print("   [bold]carto ingest ~/research ~/code && carto connect all && carto viz[/bold]")
    finally:
        if old_home is None:
            os.environ.pop("CARTOGRAPH_HOME", None)
        else:
            os.environ["CARTOGRAPH_HOME"] = old_home
        shutil.rmtree(tmp, ignore_errors=True)
