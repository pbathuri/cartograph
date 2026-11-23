"""Self-supervised relevance evaluation + tuning — how Cartograph trains itself on your real work.

Ground truth comes from AutoLog: each activity summary ("the user is profiling the SpMV kernel...") lists
the documents that were open. Map those documents onto graph projects and you get a labelled query:
"given what the user was doing, the right context is project P". Thousands of these exist, they renew
every day, and they need no manual labelling.

  build_evalset  -> queries (summary + key topics) with gold project ids, time-ordered; the most recent
                    25% is the HOLDOUT (tuning never sees it), so accepted changes must generalize forward.
  candidates     -> retrieval runs ONCE per query (FTS + semantic ranked lists); every parameter setting is
                    then scored on the cached lists, so a sweep over hundreds of settings takes seconds.
  score          -> the same math as production: RRF(k) fusion, persona field-weight steering with the
                    project-rank gate, personalized PageRank over the project graph, brief assembly with
                    per-project snippet caps. Metrics: hit@1, hit@5, MRR, brief recall (is a gold project
                    in what the agent actually receives) and brief tokens (~chars/4).
  tune           -> coordinate descent (fusion -> graph -> brief budget). Objective rewards brief recall
                    and MRR and charges for tokens. A setting is only ACCEPTED if it beats the live
                    parameters on the holdout; otherwise the live parameters stay. Every run is logged.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

from .config import Config
from .storage import Store
from .tuning import DEFAULTS

TOKEN_COST = 0.04          # objective penalty per 1k brief tokens
MIN_GAIN = 0.002           # holdout objective must improve by at least this to accept


@dataclass
class EvalQuery:
    qid: int
    ts: float
    text: str
    gold: set[int]
    fts: list[tuple[int, int, int]] = field(default_factory=list)   # (chunk_id, project_id, text_len)
    sem: list[tuple[int, int, int]] = field(default_factory=list)


def build_evalset(store: Store, *, max_queries: int = 400, autolog_db=None) -> list[EvalQuery]:
    from .graphnet import file_index, map_doc_paths, read_activity
    tail2, base = file_index(store)
    out: list[EvalQuery] = []
    seen_text: set[str] = set()
    for a in read_activity(autolog_db):                       # newest first
        gold = map_doc_paths(a["docs"], tail2, base)
        if not gold:
            continue
        text = (a["summary"][:500] + " " + " ".join(map(str, a["topics"][:8]))).strip()
        key = text[:120]
        if len(text) < 40 or key in seen_text:
            continue
        seen_text.add(key)
        out.append(EvalQuery(qid=int(a["id"]), ts=a["ts"], text=text, gold=gold))
        if len(out) >= max_queries:
            break
    out.sort(key=lambda q: q.ts)
    return out


def split(queries: list[EvalQuery], holdout: float = 0.25) -> tuple[list[EvalQuery], list[EvalQuery]]:
    n = max(1, int(len(queries) * holdout)) if queries else 0
    return queries[:-n] if n else queries, queries[-n:] if n else []


def attach_candidates(store: Store, cfg: Config, queries: list[EvalQuery], *, depth: int = 60,
                      progress=None) -> None:
    """Run retrieval once per query and cache the ranked candidate lists."""
    from .retrieve import _chunks_by_id, _semantic_chunk_ids, sanitize
    for i, q in enumerate(queries):
        san = sanitize(q.text)
        rows = store.search_chunks(san, limit=depth) if san else []
        q.fts = [(r["chunk_id"], r.get("project_id") or -1, len(r.get("chunk_text") or "")) for r in rows]
        ids = _semantic_chunk_ids(cfg, q.text, depth)
        meta = _chunks_by_id(store, ids)
        q.sem = [(cid, meta[cid].get("project_id") or -1, len(meta[cid].get("chunk_text") or ""))
                 for cid in ids if cid in meta]
        if progress and i % 50 == 0:
            progress(f"candidates {i + 1}/{len(queries)}")


def _project_fields(store: Store) -> dict[int, str]:
    with store.cursor() as c:
        return {r["id"]: (r["field"] or "general") for r in c.execute("SELECT id, field FROM projects")}


def score(queries: list[EvalQuery], params: dict, *, field_weights: dict[str, float],
          proj_field: dict[int, str], graph=None, alpha: float = 0.35, top_k: int = 6) -> dict:
    """Production-equivalent scoring of cached candidates under `params`."""
    P = {**DEFAULTS, **params}
    k = float(P["rrf_k"])
    hit1 = hit5 = rr = recall = 0.0
    tokens = 0.0
    n = 0
    for q in queries:
        if not q.gold:
            continue
        n += 1
        fused: dict[int, float] = {}
        meta: dict[int, tuple[int, int]] = {}
        for ranking in (q.sem, q.fts):
            for rank, (cid, pid, ln) in enumerate(ranking, 1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
                meta[cid] = (pid, ln)
        if not fused:
            continue
        order = sorted(fused, key=lambda c: -fused[c])[: max(top_k * 2, 12)]
        m = len(order)
        proj_rank: dict[int, int] = {}
        for cid in order:
            proj_rank.setdefault(meta[cid][0], len(proj_rank))
        scored = []
        for i, cid in enumerate(order):
            pid = meta[cid][0]
            fw = field_weights.get(proj_field.get(pid, "general"), 0.0)
            gate = max(0.0, 1.0 - proj_rank[pid] / float(P["gate_div"]))
            scored.append([1.0 - i / m + alpha * (fw - 0.5) * gate, cid, pid])
        if graph is not None and P["w_ppr"] > 0:
            seeds: dict[int, float] = {}
            for s, _, pid in scored:
                if pid >= 0:
                    seeds[pid] = max(seeds.get(pid, 0.0), max(0.0, s))
            pr = graph.ppr(seeds, damping=float(P["ppr_damping"]), prior_mix=float(P["prior_mix"]))
            mx = max(pr.values(), default=0.0)
            if mx > 0:
                for item in scored:
                    item[0] += P["w_ppr"] * pr.get(item[2], 0.0) / mx
        scored.sort(key=lambda t: -t[0])
        ranked_projects: list[int] = []
        for _, _, pid in scored:
            if pid not in ranked_projects:
                ranked_projects.append(pid)
        first = next((i for i, p in enumerate(ranked_projects) if p in q.gold), None)
        if first is not None:
            rr += 1.0 / (first + 1)
            hit1 += first == 0
            hit5 += first < 5
        per: dict[int, int] = {}
        brief_projects: set[int] = set()
        chars = 0
        taken = 0
        for _, cid, pid in scored:
            if taken >= top_k:
                break
            if per.get(pid, 0) >= int(P["snips_per_project"]):
                continue
            per[pid] = per.get(pid, 0) + 1
            taken += 1
            brief_projects.add(pid)
            chars += min(int(P["snippet_chars"]), meta[cid][1]) + 60      # + file/project labels
        recall += bool(brief_projects & q.gold)
        tokens += chars / 4.0
    if n == 0:
        return {"n": 0, "objective": 0.0}
    res = {"n": n, "hit@1": hit1 / n, "hit@5": hit5 / n, "mrr": rr / n, "brief_recall": recall / n,
           "brief_tokens": tokens / n}
    res["objective"] = res["brief_recall"] + 0.5 * res["mrr"] - TOKEN_COST * res["brief_tokens"] / 1000.0
    return {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in res.items()}


GRID = {
    "rrf_k": [15, 30, 60, 100],
    "w_ppr": [0.0, 0.1, 0.25, 0.5, 0.8],
    "prior_mix": [0.1, 0.3, 0.5],
    "ppr_damping": [0.7, 0.85],
    "gate_div": [1.5, 2.0, 3.0],
    "snips_per_project": [1, 2, 3],
    "snippet_chars": [240, 320, 400],
}
STAGES = [("rrf_k", "gate_div"), ("w_ppr", "prior_mix", "ppr_damping"), ("snips_per_project", "snippet_chars")]


def tune(train: list[EvalQuery], holdout: list[EvalQuery], live: dict, **ctx) -> dict:
    """Coordinate descent on train; accept only if the holdout objective beats the live parameters."""
    best = {**DEFAULTS, **live}
    best_train = score(train, best, **ctx)
    for stage in STAGES:
        for combo in itertools.product(*(GRID[k] for k in stage)):
            cand = {**best, **dict(zip(stage, combo))}
            if cand.get("w_ppr", 0) > 0 and ctx.get("graph") is None:
                continue
            r = score(train, cand, **ctx)
            if r["objective"] > best_train["objective"] + 1e-9:
                best, best_train = cand, r
    live_hold = score(holdout, {**DEFAULTS, **live}, **ctx)
    cand_hold = score(holdout, best, **ctx)
    accept = (cand_hold.get("objective", 0) >= live_hold.get("objective", 0) + MIN_GAIN
              and cand_hold.get("mrr", 0) >= live_hold.get("mrr", 0) - 0.01)
    return {"accepted": bool(accept), "params": best if accept else {**DEFAULTS, **live},
            "candidate": best, "train": best_train, "holdout_live": live_hold, "holdout_candidate": cand_hold}


def run_eval(store: Store, cfg: Config, *, max_queries: int = 400, do_tune: bool = True, progress=None) -> dict:
    from . import tuning
    from .graphnet import load_graph
    from .persona.profile import load_persona
    t0 = time.time()
    qs = build_evalset(store, max_queries=max_queries)
    if len(qs) < 20:
        return {"ok": False, "reason": f"only {len(qs)} labelled AutoLog queries map onto the graph (need 20)"}
    attach_candidates(store, cfg, qs, progress=progress)
    train, hold = split(qs)
    persona = load_persona(store)
    ctx = {"field_weights": persona.field_weights, "proj_field": _project_fields(store),
           "graph": load_graph(store), "alpha": getattr(persona, "learned_alpha", 0.35)}
    live = tuning.load()
    if not do_tune:
        return {"ok": True, "n": len(qs), "live": score(hold, live, **ctx)}
    res = tune(train, hold, live, **ctx)
    entry = {"n_train": len(train), "n_holdout": len(hold), "accepted": res["accepted"],
             "candidate": {k: res["candidate"][k] for k in DEFAULTS},
             "holdout_live": res["holdout_live"], "holdout_candidate": res["holdout_candidate"],
             "seconds": round(time.time() - t0, 1)}
    tuning.log_run(entry)
    if res["accepted"]:
        tuning.save(res["params"], {"holdout": res["holdout_candidate"], "train": res["train"]})
    return {"ok": True, **entry}


def dumps(d: dict) -> str:
    return json.dumps(d, indent=2, default=str)
