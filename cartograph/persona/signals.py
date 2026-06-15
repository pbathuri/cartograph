"""Preference signals — the learning loop. Each piece of feedback (the user engaged with / accepted /
got value from a result) nudges the persona: field weights move via EMA, and (if semantic is on) the
preference vector moves toward the embeddings of what they engaged with. Append-only log for audit."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import home
from .profile import PersonaProfile, save_persona


def _log_path() -> Path:
    return home() / "feedback.jsonl"


def _field_of_projects(store, names: list[str]) -> list[str]:
    if not names:
        return []
    ph = ",".join("?" * len(names))
    with store.cursor() as c:
        return [dict(r)["field"] for r in
                c.execute(f"SELECT DISTINCT field FROM projects WHERE name IN ({ph})", names).fetchall()
                if dict(r)["field"]]


def _embeddings_for_chunks(store, cfg, chunk_ids: list[int]):
    """Embed the engaged chunks' text (so the preference vector moves toward their meaning)."""
    try:
        from ..embed import DOC_PREFIX, _model, available
        if not available() or not chunk_ids:
            return None
        ph = ",".join("?" * len(chunk_ids))
        with store.cursor() as c:
            rows = c.execute(f"SELECT chunk_text FROM chunks WHERE id IN ({ph})", chunk_ids).fetchall()
        texts = [DOC_PREFIX + (dict(r)["chunk_text"] or "")[:600] for r in rows]
        if not texts:
            return None
        return _model(cfg.embed_model).encode(texts, normalize_embeddings=True)
    except Exception:
        return None


def record_feedback(profile: PersonaProfile, store, cfg, *, query: str = "",
                    liked_projects: list[str] | None = None, liked_chunks: list[int] | None = None,
                    weight: float = 1.0) -> PersonaProfile:
    """Record one preference signal and update the persona online. Returns the updated profile."""
    liked_projects = liked_projects or []
    liked_chunks = liked_chunks or []
    fields = _field_of_projects(store, liked_projects)
    for fld in fields:
        profile.bump_field(fld, amount=0.15 * weight)
    if liked_chunks:
        emb = _embeddings_for_chunks(store, cfg, liked_chunks)
        if emb is not None:
            profile.update_vector(emb, lr=min(0.4, 0.2 * weight))
    profile.n_signals += 1
    save_persona(profile)
    home().mkdir(parents=True, exist_ok=True)
    with _log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "query": query,
                            "liked_projects": liked_projects, "liked_chunks": liked_chunks,
                            "fields": fields, "weight": weight}) + "\n")
    return profile
