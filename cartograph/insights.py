"""Stats + insights for the Studio's stats window. `simple` is the at-a-glance panel; `advanced` adds
inference timings and the learned internals. Everything degrades gracefully and never raises."""
from __future__ import annotations

import time

from .config import db_path, home, load_config
from .storage import Store


def capabilities() -> dict:
    import importlib.util
    # Check availability WITHOUT importing the heavy lib (importing sentence-transformers/torch costs
    # ~8s cold) — find_spec is instant, keeping the stats panel fast.
    sem_ok = lambda: importlib.util.find_spec("sentence_transformers") is not None  # noqa: E731
    sig = 0
    try:
        from .persona.profile import load_persona
        sig = load_persona(Store(db_path(), read_only=True)).n_signals
    except Exception:
        pass
    try:
        from .security import encryption_available
        enc = encryption_available()
    except Exception:
        enc = False
    return {
        "semantic": bool(sem_ok()),
        "router_trained": (home() / "field_centroids.npz").exists(),
        "reranker_trained": (home() / "reranker.npz").exists(),
        "vision_data": Store(db_path(), read_only=True).project_id_by_name("screen-activity") is not None
        if db_path().exists() else False,
        "persona_signals": sig,
        "encryption_available": enc,
    }


def simple_stats() -> dict:
    counts = Store(db_path(), read_only=True).stats() if db_path().exists() else {}
    return {"counts": counts, "capabilities": capabilities()}


def _time_ms(fn) -> float | None:
    try:
        t = time.perf_counter()
        fn()
        return round((time.perf_counter() - t) * 1000, 1)
    except Exception:
        return None


def advanced_stats(probe: str = "how do I structure this module", deep: bool = False) -> dict:
    """Adds the learned internals + FAST inference timings (route, intent). The heavy timings (retrieve,
    brief) warm the embedding model, so they only run when `deep=True` — the panel stays instant otherwise."""
    base = simple_stats()
    cfg = load_config()
    timings: dict[str, float | None] = {}
    store = Store(db_path(), read_only=True) if db_path().exists() else None

    from .intent import classify
    from .router import route
    timings["route_ms"] = _time_ms(lambda: route(probe, cfg))
    timings["intent_ms"] = _time_ms(lambda: classify(probe))
    if deep and store is not None:
        from .retrieve import retrieve
        timings["retrieve_ms"] = _time_ms(lambda: retrieve(probe, store, cfg, top_k=5))
        try:
            from .persona import build_brief
            from .persona.profile import load_persona
            persona = load_persona(store)
            timings["brief_ms"] = _time_ms(lambda: build_brief(probe, store, cfg, persona, top_k=5))
        except Exception:
            timings["brief_ms"] = None

    internals: dict = {}
    try:
        from .persona.profile import load_persona
        p = load_persona(Store(db_path(), read_only=True))
        internals["learned_alpha"] = round(getattr(p, "learned_alpha", 0.0), 3)
        internals["top_fields"] = [{"field": f, "weight": round(w, 3)} for f, w in p.top_fields(6)]
    except Exception:
        pass
    try:
        import numpy as np
        rp = home() / "reranker.npz"
        if rp.exists():
            with np.load(rp) as z:
                names = ["base", "field_weight", "pref_cos", "field_match", "proj_affinity"]
                internals["reranker_weights"] = {n: round(float(w), 3)
                                                 for n, w in zip(names, z["w"])} if z["w"].shape[0] == 5 else {}
                internals["reranker_examples"] = int(z["n"][0]) if "n" in z else None
    except Exception:
        pass
    try:
        from .workflow import get_workflow
        wf = get_workflow()
        internals["workflow"] = {"nodes": len(wf["nodes"]), "edges": len(wf["edges"]),
                                 "custom_nodes": sum(1 for n in wf["nodes"] if not n.get("system"))}
    except Exception:
        pass

    return {**base, "timings_ms": timings, "internals": internals, "probe": probe, "deep": deep}
