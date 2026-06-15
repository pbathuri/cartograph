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
from .config import Config, db_path, home, index_dir, load_config, save_config
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
    targets = list(path) if path else cfg.roots
    if not targets:
        console.print("[red]No folders.[/red] Pass a path or run `carto init` to set roots.")
        raise typer.Exit(1)
    from .ingest import ingest_path
    store = _store()
    total = {"projects": 0, "files_indexed": 0, "chunks": 0}
    fields: dict = {}
    for t in targets:
        console.print(f"[bold]ingesting[/bold] {t} …")
        st = ingest_path(t, store, cfg, progress=lambda m: console.print(f"  {m}", style="dim"))
        total["projects"] += st.projects
        total["files_indexed"] += st.files_indexed
        total["chunks"] += st.chunks
        for k, v in st.fields.items():
            fields[k] = fields.get(k, 0) + v
    console.print(f"\n[green]✓ ingested[/green] {total['projects']} projects · "
                  f"{total['files_indexed']} files · {total['chunks']} chunks")
    if fields:
        console.print("  fields: " + ", ".join(f"{k}({v})" for k, v in sorted(fields.items(), key=lambda x: -x[1])))
    console.print("  next: [bold]carto index[/bold] for semantic search, or [bold]carto viz[/bold] to explore")


@app.command()
def index() -> None:
    """Build the semantic vector index (needs `pip install cartograph__v1[semantic]`). Re-run after ingest."""
    from .embed import available, build_index
    if not available():
        console.print(r"[yellow]semantic extra not installed.[/yellow] "
                      r"Run: [bold]pip install 'cartograph__v1\[semantic]'[/bold] (adds ~2GB; GPU-accelerated if present).")
        raise typer.Exit(1)
    cfg = load_config()
    console.print("[bold]building semantic index[/bold] (first run downloads the model) …")
    res = build_index(_store(read_only=True), cfg, progress=lambda m: console.print(f"  {m}", style="dim"))
    console.print(f"[green]✓ indexed[/green] {res.get('vectors', 0)} chunks -> {res.get('dir', index_dir())}")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="What to find."),
    top_k: int = typer.Option(8, "--top-k"),
    chunks: bool = typer.Option(False, "--chunks", help="Show snippets, not just project names."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Hybrid retrieval over your graph (semantic + keyword; semantic auto-on if indexed)."""
    from .retrieve import retrieve as _r
    res = _r(query, _store(read_only=True), load_config(), top_k=top_k)
    if json_out:
        typer.echo(json.dumps(res.to_dict(), indent=2))
        return
    console.print(f"[bold]{res.method}[/bold] — {len(res.projects)} projects")
    if chunks:
        for c in res.chunks:
            console.print(f"[cyan]{c.get('project_name','')}[/cyan] [dim]{c.get('file_path','')}[/dim]")
            console.print(f"  {(c.get('chunk_text','') or '')[:160].strip()}")
    else:
        for p in res.projects:
            console.print(f"  {p}")


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
def stats() -> None:
    """Show graph counts."""
    s = _store(read_only=True).stats()
    for k, v in s.items():
        console.print(f"  {k:10s} {v:,}")


@app.command("mcp-server")
def mcp_server() -> None:
    """Run the MCP server (stdio) so Claude Code / Cursor can query your graph. See README."""
    from .mcp_server import serve
    serve()


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
    cfg_json = json.dumps({"mcpServers": {"cartograph": {
        "command": "carto", "args": ["mcp-server"], "type": "stdio"}}}, indent=2)
    console.print("[bold]1) MCP config[/bold] — add to ~/.cursor/mcp.json (Cursor) or your Claude Code MCP settings:")
    console.print(cfg_json)
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
    console.print(f"index    : {'built' if (index_dir()/'vectors.npy').exists() else 'not built (carto index)'}")
    try:
        import torch
        console.print(f"torch    : {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception:
        console.print(r"torch    : not installed (CPU embedding only; pip install 'cartograph__v1\[ml]')")


if __name__ == "__main__":
    app()
