"""Steering — turn the persona into action. Two outputs:
  1. personalized_retrieve: hybrid retrieval re-ranked by alignment to the persona (field weights +
     preference-vector cosine), so context surfaces what fits THIS user.
  2. build_brief: a compact, model-agnostic "steering brief" any agent prepends before answering —
     persona summary, the prompt's field, top personalized snippets, and explicit output guidance.
     This is the mechanism that makes Claude/Cursor/ChatGPT/Gemini outputs adapt to the user.
"""
from __future__ import annotations

from typing import Any

from ..config import Config
from ..retrieve import retrieve
from ..storage import Store
from .profile import PersonaProfile


def _project_fields(store: Store, names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    ph = ",".join("?" * len(names))
    with store.cursor() as c:
        return {dict(r)["name"]: (dict(r)["field"] or "general")
                for r in c.execute(f"SELECT name, field FROM projects WHERE name IN ({ph})", names).fetchall()}


def personalized_retrieve(prompt: str, store: Store, cfg: Config, persona: PersonaProfile,
                          *, top_k: int = 8, alpha: float | None = None) -> dict:
    """Retrieve, then blend the base rank with a persona alignment score.
    final = (1-alpha)*base_rank_score + alpha*persona_score. Persona score = field weight of the chunk's
    project (always available) + preference-vector cosine (if a vector exists). Confidence-scaled.
    `alpha` defaults to the persona's LEARNED alpha (tuned from the record_use log)."""
    res = retrieve(prompt, store, cfg, top_k=max(top_k * 2, 12))
    chunks = res.chunks
    if not chunks:
        return {"method": res.method, "chunks": [], "projects": [], "personalized": False}
    pf = _project_fields(store, list({c.get("project_name") for c in chunks if c.get("project_name")}))
    # confidence in steering grows with how much we know about the user
    steer_conf = min(1.0, persona.n_signals / 20.0) if persona.n_signals else (0.3 if persona.field_weights else 0.0)
    base_alpha = alpha if alpha is not None else getattr(persona, "learned_alpha", 0.35)
    a = base_alpha * (0.4 + 0.6 * steer_conf)               # don't steer hard before we know the user
    vecs = persona._load_all()                              # per-field subspace vectors + _global
    from ..rerank_model import extract_features, load as load_reranker, project_affinities
    from ..router import route
    reranker = load_reranker()                             # learned model (None until `carto train` on feedback)
    rfield = route(prompt, cfg)["field"] if reranker is not None else None
    _na = lambda s: "".join(c for c in (s or "").lower() if c.isalnum())  # noqa: E731
    # Affinity is CONTEXTUAL (preference for this KIND of query) when a context model exists, else global.
    from ..context_affinity import load_context_affinity, query_vector
    ctx = load_context_affinity() if reranker is not None else None
    qv = query_vector(prompt, cfg) if ctx is not None else None
    gaff = project_affinities() if reranker is not None else {}
    ctx_on = ctx is not None and qv is not None

    def _aff(nm: str) -> tuple[float, int]:                # (affinity in [-1,1], evidence count)
        if ctx_on:
            return ctx.lookup(qv, nm)
        return (gaff.get(_na(nm), 0.0), 0)

    n = len(chunks)
    # Confidence-scaled steering: steer hard only when affinity is strong AND well-evidenced for this
    # query's context; gentle when unsure. Decoupled from the field-alpha (which floors for 1-domain users).
    ctx_conf = 0.0
    if reranker is not None:
        for ch in chunks[:4]:
            av, evd = _aff(ch.get("project_name"))
            ctx_conf = max(ctx_conf, abs(av) * (min(1.0, evd / 4.0) if ctx_on else 1.0))
    strength = (0.4 + 0.6 * steer_conf) * (0.15 + 0.85 * ctx_conf) if ctx_on else a
    # PROJECT-rank (not chunk-rank) gate: a preferred project must be promotable even when a rival's many
    # chunks fill the chunk head. Rank by first appearance among DISTINCT projects. (Chunk-rank gating
    # silently blocked promotion for multi-file projects — found in the compeng trial.)
    proj_rank: dict[str, int] = {}
    for ch in chunks:
        nm = ch.get("project_name")
        if nm and nm not in proj_rank:
            proj_rank[nm] = len(proj_rank)
    scored = []
    cd0 = False                                             # is the TOP relevance hit confidently disliked?
    for i, ch in enumerate(chunks):
        base = 1.0 - i / n                                  # base rank, 1..0
        fld = pf.get(ch.get("project_name", ""), "general")
        fw = persona.field_weights.get(fld, 0.0)            # field-weight component (always on)
        pc = 0.0
        pv = vecs.get(fld) if fld in vecs else vecs.get("_global")  # score IN the chunk's subspace
        if pv is not None:
            cv = _chunk_vec(ch, cfg)
            if cv is not None:
                import numpy as np
                pc = float(max(0.0, np.dot(pv, cv)))
        pr = proj_rank.get(ch.get("project_name"), 99)
        relgate = max(0.0, 1.0 - pr / 2.0)                  # influence only the top ~2 distinct projects
        if reranker is not None:                            # LEARNED reranker, CONTEXTUAL signal
            av, evd = _aff(ch.get("project_name"))
            feat = extract_features(base, fw, pc, 1.0 if fld == rfield else 0.0, av)
            adj = strength * (float(reranker.proba([feat])[0]) - 0.5) * relgate
            cd = ctx_on and av <= -0.4 and evd >= 2         # contextually-confident dislike FOR THIS QUERY
        else:                                               # heuristic (field weight + subspace cosine)
            ps = 0.5 * fw + 0.5 * pc if pv is not None else fw
            adj = a * (ps - 0.5) * relgate                  # centered so it re-orders, not uniformly lifts
            cd = False
        if i == 0:
            cd0 = cd
        scored.append([base + adj, ch])
    # TOP-HIT PROTECTION (covers reranker, heuristic, AND cold start): the most-relevant result keeps the
    # top score UNLESS we're contextually-confident it's the wrong project for THIS kind of query. This is
    # what makes personalization do no harm (test_v1: it stops favored projects polluting clear queries),
    # while still letting a confidently-wrong incumbent be demoted so an ambiguous query can flip.
    if scored and not cd0:
        top_other = max((s for s, _ in scored[1:]), default=scored[0][0])
        if scored[0][0] < top_other:
            scored[0][0] = top_other + 1e-6
    scored.sort(key=lambda t: t[0], reverse=True)
    out_chunks = [c for _, c in scored][:top_k]
    projects: list[str] = []
    for c in out_chunks:
        nm = c.get("project_name")
        if nm and nm not in projects:
            projects.append(nm)
    method = res.method + ("+reranker" if reranker is not None else "+persona")
    return {"method": method, "chunks": out_chunks, "projects": projects,
            "personalized": True, "steer_confidence": round(steer_conf, 3),
            "reranker": reranker is not None}


def _chunk_vec(ch: dict, cfg: Config):
    try:
        from ..embed import DOC_PREFIX, _model, available
        if not available():
            return None
        import numpy as np
        v = _model(cfg.embed_model).encode([DOC_PREFIX + (ch.get("chunk_text") or "")[:600]],
                                           normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)
    except Exception:
        return None


_DEFAULT_GUIDANCE = {
    "verbosity": "match the user's preferred verbosity",
    "tone": "technical and direct",
}


def build_brief(prompt: str, store: Store, cfg: Config, persona: PersonaProfile, *, top_k: int = 6,
                max_chars: int = 0) -> dict:
    """The model-agnostic personalization envelope an agent prepends before answering.
    `max_chars` (0 = unlimited) caps total snippet text so the brief fits a context budget."""
    from ..intent import classify
    from ..router import route
    routed = route(prompt, cfg)                      # learned field router (keyword fallback)
    pfield = routed["field"]
    intent = classify(prompt)                        # use-case type (debug/code/review/generate/explain)
    pr = personalized_retrieve(prompt, store, cfg, persona, top_k=top_k)
    prefs = {**_DEFAULT_GUIDANCE, **persona.preferences}
    guidance = []
    if persona.top_fields(1):
        tf = persona.top_fields(3)
        guidance.append("Ground the answer in the user's work; their focus is "
                        + ", ".join(f"{k}" for k, _ in tf) + ".")
    if pfield != "general":
        conf = persona.confidence.get(pfield)
        if conf:
            guidance.append(f"This prompt is in '{pfield}', a field the user works in "
                            f"(confidence {conf:.0%}) — use their conventions + the relevant repos below.")
    guidance.append(intent["guidance"])              # use-case-specific shaping (debug/code/explain/...)
    guidance.append("Output style: " + "; ".join(f"{k}={v}" for k, v in prefs.items()) + ".")
    if pr["chunks"]:
        guidance.append("Prefer the user's own patterns over generic ones; cite the files below.")
    # Live screen context (if the vision watcher is running): what the user is doing RIGHT NOW. This is
    # the honest "optimize before the output" mechanism — it enriches the INJECTED context, not the model.
    activity = []
    try:
        from ..vision.pipeline import recent_activity
        activity = recent_activity(store, limit=3)
    except Exception:
        activity = []
    if activity:
        cur = activity[0]
        where = cur.get("app") or "their screen"
        guidance.append(f"Live context: the user is currently working in {where}"
                        + (f" on a '{cur['field']}' task" if cur.get("field") else "")
                        + " — make the answer continuous with what's on screen.")
    # assemble context, optionally trimmed to a char budget (drop trailing snippets to fit)
    ctx, used, per = [], 0, 400
    if max_chars and pr["chunks"]:
        per = max(160, min(400, max_chars // max(1, len(pr["chunks"]))))
    for c in pr["chunks"]:
        snip = (c.get("chunk_text") or "")[:per]
        if max_chars and used + len(snip) > max_chars and ctx:
            break
        ctx.append({"project": c.get("project_name"), "file": c.get("file_path"), "snippet": snip})
        used += len(snip)
    return {
        "prompt": prompt,
        "prompt_field": pfield,
        "prompt_intent": intent["intent"],
        "field_routing": routed,
        "persona_summary": persona.summary(),
        "top_fields": persona.top_fields(4),
        "preferences": prefs,
        "steer_confidence": pr.get("steer_confidence", 0.0),
        "relevant_context": ctx,
        "current_activity": activity,
        "output_guidance": guidance,
        "note": "Prepend this to the model's context. Field-weight steering works with zero ML; the "
                "preference vector sharpens it once `carto index` is built and feedback accrues.",
    }
