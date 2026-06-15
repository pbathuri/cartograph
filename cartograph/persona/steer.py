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
    a = base_alpha * (0.4 + 0.6 * steer_conf)
    vecs = persona._load_all()                              # per-field subspace vectors + _global
    from ..rerank_model import extract_features, load as load_reranker, project_affinities
    from ..router import route
    reranker = load_reranker()                             # learned model (None until `carto train` on feedback)
    rfield = route(prompt, cfg)["field"] if reranker is not None else None
    aff = project_affinities() if reranker is not None else {}
    _na = lambda s: "".join(c for c in (s or "").lower() if c.isalnum())  # noqa: E731
    scored = []
    n = len(chunks)
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
        if reranker is not None:                            # LEARNED reranker decides the score
            feat = extract_features(base, fw, pc, 1.0 if fld == rfield else 0.0,
                                    aff.get(_na(ch.get("project_name")), 0.0))
            score = float(reranker.proba([feat])[0])
        else:                                               # heuristic blend (hand-tuned, learned-alpha)
            ps = 0.5 * fw + 0.5 * pc if pv is not None else fw
            score = (1 - a) * base + a * ps
        scored.append((score, ch))
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
        "output_guidance": guidance,
        "note": "Prepend this to the model's context. Field-weight steering works with zero ML; the "
                "preference vector sharpens it once `carto index` is built and feedback accrues.",
    }
