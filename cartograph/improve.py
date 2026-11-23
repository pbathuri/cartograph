"""`carto improve` — the daily self-improvement loop (run by launchd at 03:30).

  1. ingest       refresh the graph from your roots (incremental; prunes stale projects)
  2. scout        pull arXiv papers + GitHub repos for your active professional topics -> frontier cards
  3. ingest       fold the new frontier cards into the graph
  4. index        embed only new chunks (incremental, GPU when available)
  5. graph        rebuild similarity + AutoLog co-activity edges and the activity prior
  6. persona      re-derive field weights from your own work (reference + frontier folders excluded)
  7. eval + tune  self-supervised AutoLog relevance set; accept new weights only if the holdout improves
  8. report       ~/.cartograph/reports/YYYY-MM-DD.md with metrics, accepted params, topics, errors

Each step is isolated: a failure is recorded in the report and the remaining steps still run.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .config import Config, home, load_config, save_config
from .storage import Store


def _reports_dir() -> Path:
    return home() / "reports"


def ensure_frontier_root(cfg: Config) -> Config:
    from .scout import frontier_dir
    fd = str(frontier_dir())
    if fd not in cfg.roots:
        frontier_dir().mkdir(parents=True, exist_ok=True)
        cfg.roots = list(cfg.roots) + [fd]
        save_config(cfg)
    return cfg


def run(*, do_ingest: bool = True, do_scout: bool = True, do_index: bool = True, do_tune: bool = True,
        max_queries: int = 400, progress=None) -> dict:
    from .config import db_path
    log = progress or (lambda m: None)
    cfg = ensure_frontier_root(load_config())
    store = Store(db_path())
    rep: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": {}}

    def step(name, fn):
        t0 = time.time()
        try:
            out = fn()
            rep["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": out}
        except Exception as e:  # keep going; the report shows what broke
            rep["steps"][name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": repr(e),
                                  "trace": traceback.format_exc()[-1200:]}
        log(f"{name}: {'ok' if rep['steps'][name]['ok'] else 'FAILED'}")

    def ingest_all():
        from pathlib import Path as _P

        from .ingest import ingest_path
        from .vision.pipeline import PROJECT_NAME
        roots = [str(_P(r).expanduser().resolve()) for r in cfg.roots]
        pruned = store.prune_outside(roots, keep_names={PROJECT_NAME})
        tot = {"projects": 0, "files_indexed": 0, "chunks": 0, "pruned_outside": pruned}
        for r in roots:
            if _P(r).is_dir():
                st = ingest_path(r, store, cfg)
                tot["projects"] += st.projects
                tot["files_indexed"] += st.files_indexed
                tot["chunks"] += st.chunks
        return tot

    if do_ingest:
        step("ingest", ingest_all)
    if do_scout:
        def _scout():
            from .scout import scout
            pdf = Path(cfg.scout_pdf_dir).expanduser() if cfg.scout_pdf_dir else None
            if pdf is not None and not pdf.parent.exists():
                pdf = None                                       # drive not mounted: cards only
            return scout(store, cfg, pdf_dir=pdf, progress=log)
        step("scout", _scout)

        def _ingest_frontier():
            from .ingest import ingest_path
            from .scout import frontier_dir
            st = ingest_path(frontier_dir(), store, cfg)
            return {"projects": st.projects, "chunks": st.chunks}
        step("ingest_frontier", _ingest_frontier)
    if do_index:
        def _index():
            from .embed import available, build_index
            if not available():
                return {"skipped": "semantic extra not installed"}
            return build_index(store, cfg)
        step("index", _index)

    def _graph():
        from .graphnet import build_graph
        return build_graph(store, cfg, progress=log)
    step("graph", _graph)

    def _persona():
        from .persona.profile import build_from_corpus, load_persona, save_persona
        old = load_persona(store)
        p = build_from_corpus(store)
        p.preferences = old.preferences                      # keep explicit prefs + learned steering
        p.learned_alpha = old.learned_alpha
        p.n_signals = old.n_signals
        save_persona(p)
        return {"top_fields": p.top_fields(5)}
    step("persona", _persona)

    def _eval():
        from .evals import run_eval
        return run_eval(store, cfg, max_queries=max_queries, do_tune=do_tune, progress=log)
    step("eval_tune", _eval)

    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _reports_dir().mkdir(parents=True, exist_ok=True)
    path = _reports_dir() / f"{time.strftime('%Y-%m-%d')}.md"
    path.write_text(render(rep), encoding="utf-8")
    (_reports_dir() / "latest.json").write_text(json.dumps(rep, indent=2, default=str))
    rep["report"] = str(path)
    return rep


def render(rep: dict) -> str:
    L = [f"# Cartograph improve — {rep['started'][:10]}", "", f"Started {rep['started']}, finished {rep.get('finished')}.", ""]
    for name, s in rep["steps"].items():
        L.append(f"## {name} — {'ok' if s['ok'] else 'FAILED'} ({s['seconds']} s)")
        if s["ok"]:
            L.append("```json\n" + json.dumps(s["result"], indent=2, default=str)[:3000] + "\n```")
        else:
            L.append(f"Error: `{s['error']}`")
        L.append("")
    ev = rep["steps"].get("eval_tune", {}).get("result") or {}
    if ev.get("ok"):
        hl, hc = ev.get("holdout_live", {}), ev.get("holdout_candidate", {})
        L += ["## Summary", "",
              f"- Holdout queries: {ev.get('n_holdout')} (train {ev.get('n_train')})",
              f"- Live:      MRR {hl.get('mrr')}, brief recall {hl.get('brief_recall')}, tokens {hl.get('brief_tokens')}",
              f"- Candidate: MRR {hc.get('mrr')}, brief recall {hc.get('brief_recall')}, tokens {hc.get('brief_tokens')}",
              f"- Accepted: {ev.get('accepted')}"]
    return "\n".join(L) + "\n"
