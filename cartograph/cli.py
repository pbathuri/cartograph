"""`carto` — the Cartograph CLI. Plug, ingest, (optionally) embed, then retrieve / elevate / serve / view.

Typical first run:
    carto init                      # interactive setup wizard
    carto ingest ~/code             # build your graph from a folder (re-run anytime; incremental)
    carto index                     # (optional) add semantic search — needs `pip install cartograph__v1[semantic]`
    carto viz                       # open the visual graph in your browser
    carto mcp-server                # plug into Claude Code / Cursor (see README)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import typer
from rich.console import Console

# Make output bulletproof across platforms: on Windows, a piped/redirected stdout defaults to cp1252
# and crashes on '✓'/'—'/etc. Force UTF-8 with replacement so the CLI never dies on an emoji.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from . import __version__
from .config import db_path, home, index_dir, load_config, save_config
from .storage import Store

app = typer.Typer(add_completion=False, help="Cartograph — your personal cognitive graph for AI work.")
console = Console()


def _store(read_only: bool = False) -> Store:
    return Store(db_path(), read_only=read_only)


@app.command()
def version() -> None:
    """Print version + workspace location."""
    console.print(f"cartograph {__version__}")
    console.print(f"workspace: {home()}  (override with CARTOGRAPH_HOME)")


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b.|[\x00-\x1f\x7f]")


def _clean_roots(roots: list[str]) -> tuple[list[str], list[str]]:
    """Strip terminal escape junk (e.g. arrow keys typed at the prompt) and keep only real folders."""
    good, bad = [], []
    for r in roots or []:
        s = _ANSI.sub("", str(r)).strip()
        p = Path(s).expanduser() if s else None
        if p and p.is_dir():
            if str(p) not in good:
                good.append(str(p))
        else:
            bad.append(str(r))
    return good, bad


def _clean_fields(fields_in: list[str]) -> tuple[list[str], list[str]]:
    """Map free-text field names onto known field ids (case/space-insensitive, prefix match)."""
    from .ingest import FIELD_TOKENS
    known = list(FIELD_TOKENS)
    good, bad = [], []
    for f in fields_in or []:
        k = _ANSI.sub("", str(f)).strip().lower().replace(" ", "_").replace("-", "_")
        hit = k if k in known else next((x for x in known if x.startswith(k) or k in x.split("_")), None)
        if hit and hit not in good:
            good.append(hit)
        elif not hit:
            bad.append(str(f))
    return good, bad


@app.command()
def init(
    roots: list[str] = typer.Option(None, "--root", help="Folder(s) to ingest. Repeatable."),
    field: list[str] = typer.Option(None, "--field", help="Your domain(s), e.g. ml_experiment. Repeatable."),
    yes: bool = typer.Option(False, "--yes", help="Non-interactive: accept defaults / provided flags."),
) -> None:
    """Set up your workspace + config (interactive). Safe to re-run."""
    cfg = load_config()
    if roots:
        cfg.roots = [str(Path(r).expanduser()) for r in roots]
    if field:
        cfg.field_focus = list(field)
    if not yes and not roots:
        console.print("[bold]Cartograph setup[/bold] — your graph lives at "
                      f"[cyan]{home()}[/cyan] (set CARTOGRAPH_HOME to use a big/fast drive).\n")
        raw = typer.prompt("Folder(s) to ingest (comma-separated, e.g. ~/code, ~/notes)", default="")
        if raw.strip():
            cfg.roots = [str(Path(r.strip()).expanduser()) for r in raw.split(",") if r.strip()]
        fr = typer.prompt("Your field(s) (optional, comma-separated; or leave blank to auto-detect)",
                          default="")
        if fr.strip():
            cfg.field_focus = [f.strip() for f in fr.split(",") if f.strip()]
    cfg.roots, bad_roots = _clean_roots(cfg.roots)
    for r in bad_roots:
        console.print(f"[yellow]! skipped root (not a folder): {r!r}[/yellow]")
    cfg.field_focus, bad_fields = _clean_fields(cfg.field_focus)
    for f in bad_fields:
        console.print(f"[yellow]! unknown field {f!r} ignored[/yellow]")
    p = save_config(cfg)
    _store()  # create schema
    console.print(f"\n[green]✓ workspace ready[/green]  config: {p}")
    if cfg.roots:
        console.print(f"  roots: {', '.join(cfg.roots)}")
        console.print("  next: [bold]carto ingest[/bold]  (uses your configured roots)")
    else:
        console.print("  next: [bold]carto ingest <folder>[/bold]")


@app.command()
def ingest(
    path: list[str] = typer.Argument(None, help="Folder(s) to ingest. Defaults to configured roots."),
) -> None:
    """Build/refresh your graph from folders (incremental — only changed files reprocess)."""
    cfg = load_config()
    targets, missing = _clean_roots(list(path) if path else cfg.roots)
    for m in missing:
        console.print(f"[yellow]! skipping {m!r} — not a folder right now (unplugged drive?)[/yellow]")
    if not targets:
        console.print("[red]No folders.[/red] Pass a path or run `carto init` to set roots.")
        raise typer.Exit(1)
    from .ingest import ingest_path
    store = _store()
    if not path:
        # Full refresh of the configured roots: drop projects outside every root (old configs, moved
        # folders). Unmounted roots still count as roots, so an unplugged drive never loses its projects.
        all_roots = [str(Path(_ANSI.sub("", r).strip()).expanduser().resolve()) for r in cfg.roots]
        from .vision.pipeline import PROJECT_NAME as _SCREEN
        n = store.prune_outside(all_roots, keep_names={_SCREEN})
        if n:
            console.print(f"[dim]removed {n} project(s) outside your configured roots[/dim]")
    total = {"projects": 0, "files_indexed": 0, "chunks": 0, "docs": 0, "docs_unsupported": 0}
    fields: dict = {}
    for t in targets:
        console.print(f"[bold]ingesting[/bold] {t} …")
        st = ingest_path(t, store, cfg, progress=lambda m: console.print(f"  {m}", style="dim"))
        for k in total:
            total[k] += getattr(st, k)
        for k, v in st.fields.items():
            fields[k] = fields.get(k, 0) + v
    console.print(f"\n[green]✓ ingested[/green] {total['projects']} projects · "
                  f"{total['files_indexed']} files ({total['docs']} documents) · {total['chunks']} chunks")
    if total["docs_unsupported"]:
        console.print(f"  [yellow]{total['docs_unsupported']} PDFs skipped[/yellow] — no PDF reader installed: "
                      "[bold]pip install \"cartograph__v1[docs]\"[/bold]")
    if fields:
        console.print("  fields: " + ", ".join(f"{k}({v})" for k, v in sorted(fields.items(), key=lambda x: -x[1])))
    console.print("  next: [bold]carto index[/bold] for semantic search, or [bold]carto viz[/bold] to explore")


@app.command()
def index(full: bool = typer.Option(False, "--full", help="Re-embed every chunk instead of only new ones."),
          background: bool = typer.Option(False, "--background", "-b",
                                          help="Run detached (survives closing the terminal); log in ~/.cartograph/logs."),
          status: bool = typer.Option(False, "--status", help="Show progress of a running/finished build."),
          watch: bool = typer.Option(False, "--watch", help="With --status: live progress bar until it finishes."),
          ann: bool = typer.Option(False, "--ann", help="(Re)build only the ANN accelerator over the rows embedded so far.")) -> None:
    """Build the semantic vector index (needs `pip install cartograph__v1[semantic]`). Re-run after ingest;
    only chunks added since the last index are embedded, and an interrupted build resumes where it stopped.
    Past 100k chunks an ANN layer (built automatically, NumPy-only) keeps queries fast on millions of chunks."""
    from .embed import available, build_index, index_status
    if ann:
        from . import ann as A
        from .embed import _complete, building_dir
        d = index_dir() if _complete(index_dir()) else building_dir()
        if not _complete(d):
            console.print("[red]no index yet[/red] — run [bold]carto index[/bold] first")
            raise typer.Exit(1)
        with console.status("[bold]building ANN…[/bold]") as s:
            info = A.build(d, progress=lambda m: s.update(f"[bold]{m}[/bold]"))
        console.print(f"[green]✓ ANN[/green] {info}" if "rows" in info else f"[yellow]{info['skipped']}[/yellow]")
        return
    if status or watch:
        from .progress_view import show_index_status
        show_index_status(index_status, watch=watch, console=console)
        return
    if not available():
        console.print(r"[yellow]semantic extra not installed.[/yellow] "
                      r"Run: [bold]pip install 'cartograph__v1\[semantic]'[/bold] (adds ~2GB; GPU-accelerated if present).")
        raise typer.Exit(1)
    st = index_status()
    if st.get("state") == "building" and st.get("running"):
        console.print(f"[yellow]an index build is already running[/yellow] (pid {st.get('pid')}). "
                      "Watch it with [bold]carto index --status --watch[/bold].")
        raise typer.Exit(1)
    if background:
        from .background import spawn
        log = spawn(["index"] + (["--full"] if full else []), "index")
        console.print(f"[green]✓ index build started in the background[/green] — log: {log}\n"
                      "  watch: [bold]carto index --status --watch[/bold]")
        return
    cfg = load_config()
    console.print("[bold]building semantic index[/bold] (first run downloads the model) …")
    res = build_index(_store(read_only=True), cfg, full=full,
                      progress=lambda m: console.print(f"  {m}", style="dim"))
    console.print(f"[green]✓ indexed[/green] {res.get('vectors', 0)} chunks "
                  f"({res.get('embedded', 0)} embedded, {res.get('reused', 0)} reused, "
                  f"{res.get('resumed', 0)} resumed) -> {res.get('dir', index_dir())}")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="What to find."),
    top_k: int = typer.Option(8, "--top-k"),
    chunks: bool = typer.Option(False, "--chunks", help="Show snippets, not just project names."),
    mode: str = typer.Option(None, "--mode", "-m", help="mix | chunks | graph | memory | projects (budgeted bundle)"),
    max_tokens: int = typer.Option(2000, "--max-tokens", help="Hard budget for --mode output."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Hybrid retrieval over your graph (semantic + keyword; semantic auto-on if indexed)."""
    from .retrieve import retrieve as _r
    if mode:
        from .retrieve import context
        ctx = context(query, _store(read_only=True), load_config(), mode=mode, max_tokens=max_tokens,
                      top_k=top_k, cwd=os.getcwd())
        if json_out:
            typer.echo(json.dumps(ctx, indent=2))
            return
        console.print(f"[bold]{mode}[/bold] · {ctx['tokens']}/{max_tokens} tokens · {ctx.get('method', '')}")
        for m in ctx.get("memories") or []:
            console.print(f"  [magenta]{m['kind']}[/magenta] {m['text'][:160]}" + (" [yellow](stale)[/yellow]" if m.get("stale") else ""))
        for f in ctx.get("facts") or []:
            console.print(f"  [cyan]{f['entity']}[/cyan] {f['relation']} [dim]{f.get('project', '')} · "
                          f"{(f.get('evidence') or {}).get('cite', '')}[/dim]")
        for p in ctx.get("projects") or []:
            console.print(f"  [green]{p['project']}[/green] [dim]{p['hits']} hits · {p['best']}[/dim]")
        for c in ctx.get("chunks") or []:
            console.print(f"  [green]{c['cite']}[/green]\n    {c['text'][:200].strip()}")
        return
    res = _r(query, _store(read_only=True), load_config(), top_k=top_k)
    if json_out:
        typer.echo(json.dumps(res.to_dict(), indent=2))
        return
    console.print(f"[bold]{res.method}[/bold] — {len(res.projects)} projects")
    if chunks:
        for c in res.chunks:
            console.print(f"[cyan]{c.get('project_name','')}[/cyan] [dim]{c.get('cite') or c.get('file_path','')}[/dim]")
            console.print(f"  {(c.get('chunk_text','') or '')[:160].strip()}")
    else:
        for p in res.projects:
            console.print(f"  {p}")


@app.command()
def entities(
    project: str = typer.Argument(None, help="Describe one project (what it depends on, cites, links to)."),
    find: str = typer.Option(None, "--find", "-f", help="Which projects use/cite this (e.g. pydantic, 1706.03762)."),
    kind: str = typer.Option(None, "--kind", help="package | paper | repo | model | dataset"),
    limit: int = typer.Option(25, "--limit"),
    backfill: bool = typer.Option(False, "--backfill", help="Extract entities for every chunk already in the graph."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Deterministic facts from your files — dependencies, imports, papers, repos, models — with evidence."""
    from . import entities as E
    if backfill:
        store = _store()
        with console.status("[bold]reading every chunk for entities…[/bold]") as s:
            out = E.backfill(store, progress=lambda n: s.update(f"[bold]{n:,} chunks read…[/bold]"))
        console.print(f"[green]✓[/green] {out['entities']:,} entities · {out['facts']:,} facts · "
                      f"{out['links']} project links from {out['chunks']:,} chunks")
        return
    store = _store(read_only=True)
    if find:
        rows, related = E.find(store, find, kind=kind, limit=limit), []
    elif project:
        pid = store.project_id_by_name(project)
        if pid is None:
            console.print(f"[red]no project named {project!r}[/red]")
            raise typer.Exit(1)
        rows, related = E.project_entities(store, pid, kind=kind, limit=limit), E.related_projects(store, pid)
    else:
        console.print(E.counts(store))
        console.print("give a PROJECT or --find NAME (run [bold]carto entities --backfill[/bold] once on an older graph)")
        return
    if json_out:
        typer.echo(json.dumps({"facts": rows, "related": related}, indent=2))
        return
    from rich.table import Table
    t = Table(box=None, header_style="bold")
    for col in (["project"] if find else []) + ["entity", "relation", "files", "conf", "seen", "evidence"]:
        t.add_column(col, overflow="fold")
    for r in rows:
        seen = r["first_seen"] if r["first_seen"] == r["last_seen"] else f"{r['first_seen']} → {r['last_seen']}"
        cells = [r.get("project") or ""] if find else []
        ev = (r.get("evidence") or {}).get("cite", "")
        t.add_row(*cells, r["entity"], r["relation"], str(r["files"]), f"{r['confidence']:.2f}", seen or "",
                  Path(ev).name if ev else "")
    console.print(t)
    for x in related[:5]:
        console.print(f"  [magenta]↔ {x['project']}[/magenta] [dim]score {x['score']} via {', '.join(x['via'][:4])}[/dim]")


@app.command()
def elevate(
    task: str = typer.Argument(..., help="The build task."),
    project: str = typer.Option(None, "--project", help="Project path (for current grade)."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """One-shot top-of-field briefing: elite bar + reference repos + playbook + your relevant repos."""
    from .elite import elevate as _e
    e = _e(task, _store(read_only=True), load_config(), project=project)
    if json_out:
        typer.echo(json.dumps(e, indent=2))
        return
    console.print(f"[bold]{e['field']}[/bold]  grade={e['current_grade']}")
    for label, key in (("elite bar", "elite_bar"), ("reference the best", "references")):
        if e[key]:
            console.print(f"[cyan]{label}:[/cyan]")
            for x in e[key]:
                console.print(f"  - {x}")
    if e["playbook"]:
        console.print("[magenta]frontier process:[/magenta]")
        for i, s in enumerate(e["playbook"], 1):
            console.print(f"  {i}. {s}")
    if e["relevant_existing"]:
        console.print(f"[green]you already have:[/green] {', '.join(e['relevant_existing'])}")


@app.command()
def frontier(top: int = typer.Option(8, "--top"), json_out: bool = typer.Option(False, "--json")) -> None:
    """Coverage of each field's top-tier reference repos in your graph + priority backlog."""
    from .elite import frontier_report
    rep = frontier_report(_store(read_only=True), top=top)
    if json_out:
        typer.echo(json.dumps(rep, indent=2))
        return
    for f in rep["fields"]:
        tag = "" if f["active"] else " (inactive)"
        console.print(f"[bold]{f['field']}[/bold]{tag}  coverage={f['coverage_pct']}%  "
                      f"have={len(f['covered'])} gap={len(f['gaps'])}")
    if rep["priority_gaps"]:
        console.print("\n[bold]Priority gaps[/bold] (weakest frontier first):")
        for g in rep["priority_gaps"]:
            console.print(f"  GET {g['repo']}  [dim]({g['field']}: {g['teaches']})[/dim]")


@app.command()
def review(project: str = typer.Argument(..., help="Path to the build."),
           field: str = typer.Option(..., "--field", help="Field/archetype, e.g. ml_experiment."),
           json_out: bool = typer.Option(False, "--json")) -> None:
    """Grade a build against the elite Definition-of-Done (heuristic; reported vs verified)."""
    from .elite import score_build
    r = score_build(project, field)
    if json_out:
        typer.echo(json.dumps(r, indent=2))
        return
    console.print(f"[bold]{r['field']}[/bold]  grade=[cyan]{r['grade']}[/cyan]  "
                  f"met={len(r['met'])} unmet={len(r['unmet'])}")
    for u in r["unmet"]:
        console.print(f"  [yellow]gap[/yellow] {u}")
    console.print(f"[dim]{r['note']}[/dim]")


@app.command()
def train() -> None:
    """Train the in-between-layer models on YOUR data: the field router (per-field centroid embeddings).
    Re-run after big ingests. Needs the semantic extra; degrades to keyword routing without it."""
    from .router import build_centroids
    from .embed import available
    if not available():
        console.print(r"[yellow]semantic extra not installed[/yellow] — routing uses keywords. "
                      r"For the learned router: [bold]pip install 'cartograph__v1\[semantic]'[/bold] then carto index.")
        raise typer.Exit(1)
    console.print("[bold]training field router[/bold] (centroid per field from your corpus) …")
    counts = build_centroids(_store(read_only=True), load_config(),
                             progress=lambda m: console.print(f"  {m}", style="dim"))
    if counts:
        console.print("[green]✓ router trained[/green] on " + ", ".join(f"{k}({v})" for k, v in counts.items()))
    else:
        console.print("[yellow]no field-labeled chunks[/yellow] for the router — run carto ingest (declare --field).")
    # R4: learned/clustered fields — relabel projects the keyword inference left as 'general' (non-dev
    # corpora) with emergent clusters, so field weights + subspaces work for any vocabulary.
    from .learned_fields import learn_fields
    from .persona.profile import build_from_corpus, load_persona as _lp, save_persona as _sp
    lf = learn_fields(_store(), load_config())
    if lf.get("learned"):
        console.print(f"[green]✓ learned fields[/green]: {lf['clusters']} clusters over {lf['projects']} "
                      f"unfielded projects → {sorted(set(lf['fields'].values()))}")
        old = _lp(_store(read_only=True))
        fresh = build_from_corpus(_store(read_only=True))
        old.field_weights = fresh.field_weights                # refresh weights for the learned fields...
        for k, v in fresh.confidence.items():
            old.confidence.setdefault(k, v)
        _sp(old)                                               # ...while keeping learned_alpha/prefs/signals
    # contextual affinity: cluster the feedback queries so preference is per query-context (must be built
    # BEFORE the reranker, which consumes it as a feature). Quietly skips without enough feedback.
    from .context_affinity import build_contexts
    crep = build_contexts(load_config())
    if crep.get("trained"):
        console.print(f"[green]✓ contextual affinity[/green]: {crep['clusters']} query-contexts "
                      f"from {crep['events']} events {crep['cluster_sizes']}")
    # learned reranker from the feedback log (self-improving; activates once enough signals exist)
    from .rerank_model import train_from_log
    from .persona.profile import load_persona
    store = _store(read_only=True)
    rep = train_from_log(store, load_config(), load_persona(store))
    if rep.get("trained"):
        console.print(f"[green]✓ reranker trained[/green] on {rep['examples']} labeled candidates "
                      f"from {rep['events']} feedback events (fit {rep['fit_accuracy']:.0%} on your history)")
        console.print(f"  learned weights: {rep['weights']}")
    else:
        console.print(f"[dim]reranker: {rep.get('reason')} — give feedback (carto feedback / MCP record_use), "
                      f"then re-run carto train[/dim]")


@app.command()
def fields(json_out: bool = typer.Option(False, "--json")) -> None:
    """Learn emergent fields by clustering your corpus and relabel projects left as 'general' (helps
    non-dev users whose vocabulary doesn't match the built-in field keywords). Needs the semantic index."""
    from .learned_fields import learn_fields
    rep = learn_fields(_store(), load_config())
    if json_out:
        typer.echo(json.dumps(rep, indent=2))
        return
    if rep.get("learned"):
        console.print(f"[green]✓ learned {rep['clusters']} fields[/green] over {rep['projects']} projects:")
        for proj, fld in sorted(rep["fields"].items()):
            console.print(f"  {proj:28s} → [cyan]{fld}[/cyan]")
        console.print("[dim]run `carto persona --rebuild` to refresh field weights.[/dim]")
    else:
        console.print(f"[yellow]not learned[/yellow]: {rep.get('reason')}")


@app.command()
def route(prompt: str = typer.Argument(..., help="A prompt to route to its field."),
          json_out: bool = typer.Option(False, "--json")) -> None:
    """Show which field a prompt routes to (learned router if trained, else keywords)."""
    from .router import route as _route
    r = _route(prompt, load_config())
    if json_out:
        typer.echo(json.dumps(r, indent=2))
        return
    extra = f"  (score {r['score']}, vs {r['runner_up']})" if r.get("score") is not None else ""
    console.print(f"[bold]{r['field']}[/bold]  [dim]via {r['method']}{extra}[/dim]")


@app.command()
def watch(interval: float = typer.Option(60.0, "--interval", help="Seconds between captures."),
          minutes: float = typer.Option(0.0, "--minutes", help="Stop after N minutes (0 = run until Ctrl-C)."),
          apply: bool = typer.Option(False, "--apply", help="Actually store + learn. Default is a DRY RUN."),
          no_redact: bool = typer.Option(False, "--no-redact", help="(not recommended) skip secret redaction")) -> None:
    """Real-time vision: periodically screenshot -> novelty-cache -> OCR -> redact -> classify -> graph.

    PRIVACY: local-only; sensitive windows (banking/passwords/auth) are skipped; secrets are redacted
    before anything is stored. Default is a DRY RUN (nothing is saved) so you can see exactly what would
    be captured. Add --apply to enable. Touch ~/.cartograph/vision.paused to pause; remove it to resume.
    """
    from .vision.capture import default_capturer
    from .vision.ocr import default_ocr
    from .vision.pipeline import VisionConfig
    from .vision.watch import watch as _watch

    cap, ocr = default_capturer(), default_ocr()
    if cap is None or ocr is None:
        missing = []
        if cap is None:
            missing.append("screen capture (mss + Pillow)")
        if ocr is None:
            missing.append("OCR (pytesseract + the Tesseract binary)")
        console.print("[red]Vision needs extra dependencies:[/red] " + ", ".join(missing))
        console.print("Install with [bold]pip install 'cartograph__v1\\[vision]'[/bold] "
                      "(and the Tesseract engine for OCR), then re-run.")
        raise typer.Exit(1)

    vcfg = VisionConfig(interval_sec=interval, redact=not no_redact)
    persona = None
    if apply:
        from .persona.profile import load_persona
        persona = load_persona(_store(read_only=True))
    iters = int(minutes * 60 / interval) if minutes > 0 else None
    mode = "[green]APPLY[/green] (storing + learning)" if apply else "[yellow]DRY RUN[/yellow] (nothing saved)"
    console.print(f"[bold]carto watch[/bold] — {mode}, every {interval:g}s. "
                  + (f"stopping after {minutes:g} min." if iters else "Ctrl-C to stop."))
    console.print("[dim]Sensitive windows are skipped; secrets redacted before storage.[/dim]\n")

    def show(rec: dict) -> None:
        if rec["action"] in ("store", "preview"):
            r = (f"  redacted {rec['redacted']}" if rec.get("redacted") else "")
            console.print(f"[green]●[/green] {rec['action']:7s} field=[cyan]{rec['field']}[/cyan] "
                          f"intent={rec['intent']} chars={rec['chars']}{r}  [dim]{rec.get('app','')[:40]}[/dim]")
        else:
            console.print(f"[dim]· skip ({rec['reason']}) {rec.get('app','')[:40]}[/dim]")
    try:
        summary = _watch(_store(), load_config(), vcfg, cap, ocr, iterations=iters,
                         apply=apply, persona=persona, on_record=show)
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped.[/yellow]")
        return
    console.print(f"\n[bold]done[/bold] — {summary['ticks']} ticks: "
                  + ", ".join(f"{k}×{v}" for k, v in summary["counts"].items()))


@app.command()
def persona(rebuild: bool = typer.Option(False, "--rebuild", help="Re-derive field weights from the corpus."),
            json_out: bool = typer.Option(False, "--json")) -> None:
    """Show (or rebuild) your learned persona: field weights, preferences, confidence."""
    from .persona.profile import build_from_corpus, load_persona, save_persona
    store = _store(read_only=True)
    p = build_from_corpus(store) if rebuild else load_persona(store)
    if rebuild:
        save_persona(p)
    if json_out:
        typer.echo(json.dumps(p.to_dict(), indent=2))
        return
    console.print(f"[bold]persona[/bold] — {p.summary()}")
    for f, w in p.top_fields(8):
        bar = "█" * int(w * 30)
        console.print(f"  {f:16s} {w:6.1%} {bar}  [dim]conf {p.confidence.get(f, 0):.0%}[/dim]")


@app.command()
def personalize(prompt: str = typer.Argument(..., help="The prompt your agent is about to answer."),
                budget: int = typer.Option(0, "--budget", help="Max chars of snippet context (0 = unlimited)."),
                json_out: bool = typer.Option(False, "--json")) -> None:
    """Emit the steering brief for a prompt — the personalization envelope an agent prepends."""
    from .persona import build_brief
    from .persona.profile import load_persona
    store = _store(read_only=True)
    brief = build_brief(prompt, store, load_config(), load_persona(store), max_chars=budget)
    if json_out:
        typer.echo(json.dumps(brief, indent=2))
        return
    console.print(f"[bold]prompt field:[/bold] {brief['prompt_field']} [dim](via {brief['field_routing']['method']})[/dim]  "
                  f"[bold]intent:[/bold] {brief['prompt_intent']}  "
                  f"[dim](steer confidence {brief['steer_confidence']:.0%})[/dim]")
    console.print(f"[cyan]persona:[/cyan] {brief['persona_summary']}")
    console.print("[magenta]output guidance:[/magenta]")
    for g in brief["output_guidance"]:
        console.print(f"  - {g}")
    if brief["relevant_context"]:
        console.print("[green]ground in:[/green]")
        for c in brief["relevant_context"][:4]:
            console.print(f"  {c['project']}  [dim]{c['file']}[/dim]")


@app.command()
def feedback(query: str = typer.Option("", "--query", help="The query this feedback is about."),
             liked: list[str] = typer.Option(None, "--liked", help="Project(s) that helped. Repeatable."),
             disliked: list[str] = typer.Option(None, "--disliked", help="Project(s) that didn't. Repeatable."),
             weight: float = typer.Option(1.0, "--weight")) -> None:
    """Record a preference signal — teaches the persona what you respond to MORE/LESS (adapts over time)."""
    from .persona import record_feedback
    from .persona.profile import load_persona
    store = _store(read_only=True)
    p = record_feedback(load_persona(store), store, load_config(), query=query,
                        liked_projects=list(liked or []), disliked_projects=list(disliked or []),
                        weight=weight)
    console.print(f"[green]✓ recorded[/green] (signal #{p.n_signals}). persona: {p.summary()}")


@app.command()
def prefs(set_: list[str] = typer.Option(None, "--set", help="Output preference key=value. Repeatable, "
                                         "e.g. --set verbosity=concise --set tone=friendly --set format=bullets."),
          clear: bool = typer.Option(False, "--clear", help="Remove all explicit preferences."),
          json_out: bool = typer.Option(False, "--json")) -> None:
    """Set explicit OUTPUT-TUNING preferences (verbosity/tone/format/...) that steer every answer."""
    from .persona.profile import load_persona, save_persona
    store = _store(read_only=True)
    p = load_persona(store)
    if clear:
        p.preferences = {}
    for kv in (set_ or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            p.preferences[k.strip()] = v.strip()
    save_persona(p)
    if json_out:
        typer.echo(json.dumps(p.preferences, indent=2))
        return
    console.print("[bold]output preferences[/bold] (applied to every steering brief):")
    for k, v in (p.preferences or {"(none set)": "defaults apply"}).items():
        console.print(f"  {k} = {v}")


@app.command()
def compass(
    sync: bool = typer.Option(False, "--sync", help="Fold new AutoLog summaries into the constellation."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop the map and sync again from the first summary."),
    prompt: str = typer.Option("", "--prompt", help="Mark which directions this prompt touches."),
    db: str = typer.Option("", "--db", help="ContextD sqlite path. Default: the AutoLog database."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Map of ambitions, ventures, inquiries, sparks, and temporary sorties.

    Built so a model can hold many evolving directions at once. Quiet threads stay
    in memory. Temporary work leaves a residue instead of becoming an identity.
    Entertainment never becomes a direction. Sync reads AutoLog; the map is local.
    """
    from pathlib import Path

    from .compass import compass_for_prompt, constellation, render_text, reset as reset_compass, sync as sync_compass
    store = _store(read_only=not (sync or rebuild))
    if rebuild:
        reset_compass(store)
        sync = True
    if sync:
        report = sync_compass(store, db=Path(db) if db else None, load_defaults=True)
        console.print(
            f"[green]✓ compass[/green] {report['directions']} directions · "
            f"linked {report['observations_linked']} · "
            f"{report.get('log_nodes', 0)} log nodes · "
            f"{report.get('log_edges', 0)} reasoning edges · "
            f"kept {report['life_skipped']} non-work summaries out"
        )
    data = compass_for_prompt(store, prompt) if prompt else constellation(store)
    if json_out:
        typer.echo(json.dumps(data, indent=2))
        return
    console.print(render_text(data))


@app.command()
def stats() -> None:
    """Show graph counts."""
    s = _store(read_only=True).stats()
    for k, v in s.items():
        console.print(f"  {k:10s} {v:,}")


@app.command()
def improve(
    no_ingest: bool = typer.Option(False, "--no-ingest"), no_scout: bool = typer.Option(False, "--no-scout"),
    no_index: bool = typer.Option(False, "--no-index"), no_tune: bool = typer.Option(False, "--no-tune"),
    queries: int = typer.Option(400, "--queries", help="AutoLog eval queries to use."),
    rollback: bool = typer.Option(False, "--rollback", help="Restore the parameters before the last accepted tune."),
) -> None:
    """Daily self-improvement: ingest -> scout -> index -> graph -> persona -> eval+tune -> report."""
    if rollback:
        from .tuning import rollback as _rb
        prev = _rb()
        console.print(f"[green]restored[/green] {prev}" if prev else "[yellow]nothing to roll back[/yellow]")
        return
    from .improve import run
    rep = run(do_ingest=not no_ingest, do_scout=not no_scout, do_index=not no_index, do_tune=not no_tune,
              max_queries=queries, progress=lambda m: console.print(f"  {m}", style="dim"))
    ev = rep["steps"].get("eval_tune", {}).get("result") or {}
    if ev.get("ok"):
        console.print(f"holdout MRR {ev['holdout_live'].get('mrr')} -> {ev['holdout_candidate'].get('mrr')}, "
                      f"brief recall {ev['holdout_live'].get('brief_recall')} -> {ev['holdout_candidate'].get('brief_recall')}, "
                      f"tokens {ev['holdout_live'].get('brief_tokens')} -> {ev['holdout_candidate'].get('brief_tokens')} "
                      f"| accepted: {ev['accepted']}")
    console.print(f"report: {rep['report']}")


@app.command()
def graph() -> None:
    """Rebuild the project graph (semantic similarity + AutoLog co-activity edges, activity prior)."""
    from .graphnet import build_graph
    console.print(json.dumps(build_graph(_store(), load_config(), progress=lambda m: console.print(f"  {m}", style="dim")), indent=2))


@app.command()
def scout(topic: list[str] = typer.Option(None, "--topic", help="Scout these instead of auto topics."),
          dry_run: bool = typer.Option(False, "--dry-run", help="Only show the topics that would be searched.")) -> None:
    """Pull arXiv papers + GitHub repos for your active professional topics into ~/.cartograph/frontier."""
    from .scout import gather_topics, scout as _scout
    cfg, store = load_config(), _store()
    topics = list(topic) if topic else gather_topics(store, cfg)
    if dry_run:
        console.print("\n".join(topics) or "(no topics)")
        return
    pdf = Path(cfg.scout_pdf_dir).expanduser() if cfg.scout_pdf_dir else None
    console.print(json.dumps(_scout(store, cfg, topics=topics, pdf_dir=pdf), indent=2))


@app.command("eval")
def eval_cmd(queries: int = typer.Option(300, "--queries"), tune: bool = typer.Option(False, "--tune")) -> None:
    """Score retrieval on the self-supervised AutoLog relevance set (optionally tune + accept if better)."""
    from .evals import dumps, run_eval
    console.print(dumps(run_eval(_store(), load_config(), max_queries=queries, do_tune=tune,
                                 progress=lambda m: console.print(f"  {m}", style="dim"))))


@app.command()
def bench(suite: list[str] = typer.Option(["citations", "scifact"], "--suite",
                                          help="citations (bundled PDF/DOCX/PPTX/MD/PY corpus) and/or scifact (BEIR)."),
          semantic: bool = typer.Option(False, "--semantic", help="Also embed and score hybrid retrieval."),
          limit: int = typer.Option(0, "--limit", help="Only the first N SciFact queries (0 = all)."),
          as_json: bool = typer.Option(False, "--json")) -> None:
    """Reproducible benchmarks in a throwaway home (never touches your graph); prints a Markdown table."""
    from .bench import markdown, run
    rep = run(list(suite), semantic=semantic, limit=limit or None,
              progress=lambda m: console.print(f"  {m}", style="dim"))
    if as_json:
        console.print_json(json.dumps(rep))
    else:
        console.print(markdown(rep), markup=False, soft_wrap=True)
        console.print(f"results: {rep['path']}", style="dim")


@app.command("mcp-server")
def mcp_server(read_only: bool = typer.Option(False, "--read-only",
                                              help="Expose only tools that never write (hides feedback/memory tools).")
               ) -> None:
    """Run the MCP server (stdio) so Claude Code / Cursor / Codex / any MCP client can query your graph."""
    from .mcp_server import serve
    serve(read_only_mode=read_only)


@app.command()
def viz(port: int = typer.Option(8787, "--port"), no_open: bool = typer.Option(False, "--no-open")) -> None:
    """Launch the interactive visual graph in your browser."""
    from .viz.app import launch
    launch(port=port, open_browser=not no_open)


@app.command()
def serve(port: int = typer.Option(8787, "--port")) -> None:
    """Run the local API (no browser) — exposes /api/personalize for browser userscripts (web GenAI)."""
    from .viz.app import launch
    console.print(f"Cartograph API on http://127.0.0.1:{port}  — GET /api/personalize?prompt=...")
    launch(port=port, open_browser=False)


@app.command()
def studio(port: int = typer.Option(8787, "--port"), no_open: bool = typer.Option(False, "--no-open")) -> None:
    """Launch the visual Studio: see + edit your workflow (add/connect/import nodes), with the core
    locked behind disclaimers, a stats window, node details, and a graph chat. Non-technical friendly."""
    from .viz.app import launch
    console.print(f"[bold]Cartograph Studio[/bold] on http://127.0.0.1:{port}/studio "
                  "[dim](local-only; mutations require the workspace token, injected into the page)[/dim]")
    launch(port=port, open_browser=not no_open, path="/studio")


@app.command()
def secure() -> None:
    """Show the security posture of this workspace (token, encryption, file perms) — V1 hardening."""
    from .security import _key_path, _token_path, api_token, encryption_available
    api_token()                                          # ensure it exists
    console.print("[bold]security[/bold]")
    console.print(f"  api token   : present \\[{_token_path()}] — required for Studio mutations")
    console.print(f"  encryption  : {'available' if encryption_available() else 'off'} "
                  + ("" if encryption_available() else r"(pip install 'cartograph__v1\[secure]' for at-rest encryption)"))
    console.print(f"  enc key     : {'present' if _key_path().exists() else 'not created (used only if you enable encrypt_at_rest)'}")
    console.print("  server      : binds 127.0.0.1 only · CORS allowlist (no '*') · DNS-rebinding Host check")
    console.print("  vision      : sensitive windows skipped · secrets redacted pre-storage · opt-in encrypt-at-rest")


_AGENT_RULE = (
    "At the START of every task, call the Cartograph MCP tool `personalize` with the user's request; "
    "follow its `output_guidance` and prefer the user's own patterns in `relevant_context`. For grounding "
    "in their code/docs, call `retrieve_context`. For a frontier-grade plan, call `elevate_task`. After "
    "you answer using Cartograph context, call `record_use` with the projects that helped — so it adapts."
)


@app.command("agent-setup")
def agent_setup(rules_file: bool = typer.Option(False, "--write-rules",
                help="Also write the agent rule to .cartograph_agent_rules.md in the cwd.")) -> None:
    """Print the exact MCP config + the system-prompt rule that makes any agent use Cartograph on every
    prompt. Paste the JSON into your MCP client and the rule into your agent's system prompt / rules."""
    from .connect import server
    cfg_json = json.dumps({"mcpServers": {"cartograph": server()}}, indent=2)
    console.print("[bold]1) MCP config[/bold] — easiest: [bold]carto connect all[/bold] (writes it for every client "
                  "found). Or paste this into your client's MCP settings:")
    console.print(cfg_json, markup=False, soft_wrap=True)
    console.print("\n[bold]2) System-prompt / rules[/bold] — paste into your agent's rules so it auto-uses Cartograph:")
    console.print(f"[dim]{_AGENT_RULE}[/dim]")
    console.print("\n[bold]3) Web GenAI[/bold] (ChatGPT/Gemini): run [bold]carto serve[/bold] + the userscript in docs/BROWSER.md.")
    if rules_file:
        out = Path(".cartograph_agent_rules.md")
        out.write_text(f"# Cartograph agent rule\n\n{_AGENT_RULE}\n", encoding="utf-8")
        console.print(f"\n[green]✓ wrote[/green] {out.resolve()}")


@app.command()
def demo() -> None:
    """See everything work in ~10s on a synthetic corpus — zero setup, your real workspace untouched."""
    from .demo import run_demo
    console.print("[bold]Cartograph demo[/bold] — building a graph + steering answers on sample data.")
    run_demo(console)


@app.command()
def doctor() -> None:
    """Check your install + workspace (what's enabled, what to install for more)."""
    from .embed import available as sem_ok
    console.print(f"cartograph {__version__}")
    console.print(f"workspace: {home()}")
    console.print(f"graph db : {'present' if db_path().exists() else 'not built (run carto ingest)'}")
    console.print(r"semantic : " + ("enabled" if sem_ok() else r"off — pip install 'cartograph__v1\[semantic]'"))
    from .embed import index_status
    ist = index_status()
    console.print("index    : " + {"complete": "built",
                                   "building": f"building {ist.get('done', 0):,}/{ist.get('total', 0):,}"
                                               + ("" if ist.get("running") else " (paused — `carto index` resumes)"),
                                   }.get(ist["state"], "not built (carto index)"))
    try:
        import torch
        console.print(f"torch    : {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception:
        console.print(r"torch    : not installed (CPU embedding only; pip install 'cartograph__v1\[ml]')")


memory_app = typer.Typer(help="Agent memory shared by every client: decisions, rejected approaches, todos, sessions.")
app.add_typer(memory_app, name="memory")


def _print_memories(items: list[dict]) -> None:
    for m in items:
        closed = f" [dim](until {m['valid_to']} → {m['superseded_by']})[/dim]" if m.get("valid_to") else ""
        stale = " [yellow](possibly stale)[/yellow]" if m.get("stale") else ""
        console.print(f"[bold]{m['id']:>5}[/bold] [cyan]{m['kind']:<10}[/cyan] {m['text'][:200]}{stale}{closed}")
        console.print(f"        [dim]{m.get('project') or 'global'} · {m.get('source')} · {m['created']}[/dim]")
        for s in m.get("stale") or []:
            console.print(f"        [yellow]{s}[/yellow]")


@memory_app.command("add")
def memory_add(
    text: str = typer.Argument(..., help="What to remember."),
    kind: str = typer.Option("note", "--kind", "-k", help="decision | rejected | preference | fact | todo | note"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (default: the project at the cwd)."),
    anchor: list[str] = typer.Option(None, "--anchor", "-a", help="File this depends on (flags staleness on change)."),
    supersedes: list[int] = typer.Option(None, "--supersedes", "-s", help="Memory id(s) this replaces."),
    global_: bool = typer.Option(False, "--global", help="Not tied to any project."),
) -> None:
    """Remember something (a decision, a rejected approach, a todo...)."""
    from . import memory as M
    m = M.remember(_store(), text, kind, project=project, cwd=None if global_ else os.getcwd(),
                   anchors=anchor or [], supersedes=supersedes or [])
    console.print(f"[green]✓[/green] remembered [bold]{m['id']}[/bold] ({m['kind']}, {m.get('project') or 'global'})"
                  + (" — already known" if m.get("duplicate") else ""))
    for c in m.get("possible_conflicts") or []:
        console.print(f"  [yellow]similar to {c['id']}[/yellow] ({c['kind']}): {c['text'][:120]}  "
                      f"[dim]→ carto memory supersede {c['id']} --by {m['id']}[/dim]")


@memory_app.command("recall")
def memory_recall(
    query: str = typer.Argument("", help="Keywords (empty = newest)."),
    project: str = typer.Option(None, "--project", "-p"),
    kind: list[str] = typer.Option(None, "--kind", "-k"),
    history: bool = typer.Option(False, "--history", help="Include superseded memories."),
    limit: int = typer.Option(15, "--limit"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Find memories (active only unless --history), with staleness warnings."""
    from . import memory as M
    res = M.recall(_store(), query, project=project, kinds=kind or None, history=history, limit=limit)
    if json_out:
        typer.echo(json.dumps(res, indent=2))
        return
    _print_memories(res["memories"]) if res["memories"] else console.print("[dim]nothing remembered yet[/dim]")


@memory_app.command("supersede")
def memory_supersede(
    old: int = typer.Argument(..., help="Memory id to close."),
    by: int = typer.Option(None, "--by", help="The memory that replaces it."),
    reason: str = typer.Option("", "--reason"),
) -> None:
    """Close a memory (it stays in history with its validity interval)."""
    from . import memory as M
    M.supersede(_store(), old, by, reason)
    console.print(f"[green]✓[/green] {old} superseded" + (f" by {by}" if by else ""))


@memory_app.command("forget")
def memory_forget(mid: int = typer.Argument(...)) -> None:
    """Delete a memory permanently (prefer `supersede`, which keeps history)."""
    from . import memory as M
    console.print("[green]✓ deleted[/green]" if M.forget(_store(), mid) else f"[red]no memory {mid}[/red]")


@memory_app.command("brief")
def memory_brief(
    project: str = typer.Option(None, "--project", "-p"),
    max_chars: int = typer.Option(2000, "--max-chars"),
) -> None:
    """The session-start digest an agent receives for this folder's project."""
    from . import memory as M
    b = M.brief(_store(read_only=True), project=project, cwd=os.getcwd(), max_chars=max_chars)
    console.print(b["text"] or "[dim]no memories for this project yet[/dim]")


@memory_app.command("hook", hidden=True)
def memory_hook(client: str = typer.Option(..., "--client"),
                payload: str = typer.Argument(None, help="Codex passes the event as an argument.")) -> None:
    """Hook entry point for agent clients (reads the event JSON on stdin)."""
    from .hooks import run_hook
    raw = payload if payload is not None else ("" if sys.stdin.isatty() else sys.stdin.read())
    sys.stdout.write(run_hook(client, raw) + "\n")


hooks_app = typer.Typer(help="Capture agent sessions automatically (Claude Code, Cursor, Gemini CLI, Codex).")
app.add_typer(hooks_app, name="hooks")


@hooks_app.command("install")
def hooks_install(
    client: list[str] = typer.Argument(None, help="claude-code | cursor | gemini | codex (default: all found)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Add Cartograph's session hooks to your clients' user config (backs up the file first)."""
    from . import hooks as H
    targets = client or [s["client"] for s in H.status() if s["exists"] or s["client"] == "cursor"]
    for c in targets:
        if c not in H.CLIENTS:
            console.print(f"[red]unknown client {c}[/red] (choose from {', '.join(H.CLIENTS)})")
            continue
        r = H.install(c, dry_run=dry_run)
        mark = "[green]✓[/green]" if r["changed"] else "[dim]·[/dim]"
        console.print(f"{mark} {c:<12} {r['path']}  {', '.join(r.get('events') or []) or r.get('note', 'already installed')}")
    if not dry_run:
        console.print("[dim]restart the client (or reload its window) to pick up the hooks[/dim]")


@hooks_app.command("uninstall")
def hooks_uninstall(client: list[str] = typer.Argument(None)) -> None:
    """Remove Cartograph's hook entries (other hooks are left alone)."""
    from . import hooks as H
    for c in client or list(H.CLIENTS):
        r = H.uninstall(c)
        console.print(f"{'[green]✓ removed[/green]' if r['changed'] else '[dim]· none[/dim]'} {c}")


@hooks_app.command("status")
def hooks_status() -> None:
    """Which clients have Cartograph's session hooks."""
    from . import hooks as H
    for s in H.status():
        state = "[green]installed[/green]" if s["installed"] else ("not installed" if s["exists"] else "[dim]client not found[/dim]")
        console.print(f"  {s['client']:<12} {state}  [dim]{s['config']}[/dim]")


@app.command()
def connect(
    client: list[str] = typer.Argument(None, help="claude-code, claude-desktop, cursor, windsurf, vscode, cline, "
                                                  "gemini, codex, zed, continue, lm-studio, goose — or 'all' (found ones)"),
    read_only: bool = typer.Option(False, "--read-only", help="Serve only tools that never write (no memory writes)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change; write nothing."),
    show: bool = typer.Option(False, "--print", help="Print the config snippet to paste by hand instead."),
    remove: bool = typer.Option(False, "--remove", help="Remove Cartograph from these clients."),
) -> None:
    """Plug Cartograph's MCP server into your AI clients (edits their config; backs it up first).
    With no client: show which clients are installed and connected."""
    from . import connect as C
    if not client:
        from rich.table import Table
        t = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
        for col in ("client", "status", "config"):
            t.add_column(col, overflow="fold")
        for s in C.status():
            state = ("[green]● connected[/green]" if s["connected"] else f"[yellow]○[/yellow] carto connect {s['client']}") \
                if s["detected"] else "[dim]· not installed[/dim]"
            home = str(Path.home())
            t.add_row(s["label"], state, f"[dim]{s['path'].replace(home, '~', 1)}[/dim]")
        console.print(t)
        return
    targets = [s["client"] for s in C.status() if s["detected"]] if client == ["all"] else list(client)
    for c in targets:
        if c not in C.CLIENTS:
            console.print(f"[red]unknown client {c}[/red] (choose from {', '.join(C.CLIENTS)})")
            continue
        if show:
            console.print(f"[bold]{C.CLIENTS[c][0]}[/bold]  [dim]{C.config_path(c)}[/dim]")
            console.print(C.snippet(c, read_only), markup=False, soft_wrap=True)
            continue
        r = C.disconnect(c) if remove else C.connect(c, read_only=read_only, dry_run=dry_run)
        mark = "[green]✓[/green]" if r["changed"] else "[dim]·[/dim]"
        console.print(f"{mark} {r['label']:<30} {r['path']}  {r.get('note') or ('removed' if remove else 'connected')}")
        if r.get("snippet"):
            console.print(r["snippet"], markup=False, soft_wrap=True)
    if targets and not (dry_run or show):
        console.print("[dim]restart the client to load the server. Session capture is separate: carto hooks install[/dim]")


if __name__ == "__main__":
    app()
