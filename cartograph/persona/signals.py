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


def _embeddings_by_field(store, cfg, chunk_ids: list[int]) -> dict[str, list]:
    """Embed engaged chunks and group their vectors by the field of the owning project — so each
    field's SUBSPACE vector moves toward the meaning of what the user engaged with in that field."""
    try:
        from ..embed import DOC_PREFIX, _model, available
        if not available() or not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        with store.cursor() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT ch.id AS id, ch.chunk_text AS t, p.field AS field FROM chunks ch "
                "JOIN files f ON f.id=ch.file_id LEFT JOIN projects p ON p.id=f.project_id "
                f"WHERE ch.id IN ({ph})", chunk_ids).fetchall()]
        if not rows:
            return {}
        embs = _model(cfg.embed_model).encode([DOC_PREFIX + (r["t"] or "")[:600] for r in rows],
                                              normalize_embeddings=True)
        out: dict[str, list] = {}
        for r, e in zip(rows, embs):
            out.setdefault(r["field"] or "general", []).append(e)
        return out
    except Exception:
        return {}


def record_feedback(profile: PersonaProfile, store, cfg, *, query: str = "",
                    liked_projects: list[str] | None = None, liked_chunks: list[int] | None = None,
                    disliked_projects: list[str] | None = None, disliked_chunks: list[int] | None = None,
                    weight: float = 1.0) -> PersonaProfile:
    """Record one preference signal and update the persona online (bidirectional). 'liked' moves the
    persona toward it; 'disliked' moves it away — the full 'more/less response' loop. Returns the profile."""
    liked_projects = liked_projects or []
    liked_chunks = liked_chunks or []
    disliked_projects = disliked_projects or []
    disliked_chunks = disliked_chunks or []
    profile.decay()                                                # recency: old emphasis fades a touch
    up = _field_of_projects(store, liked_projects)
    down = _field_of_projects(store, disliked_projects)
    for fld in up:
        profile.bump_field(fld, amount=0.15 * weight)
    for fld in down:
        profile.bump_field(fld, amount=-0.10 * weight)            # pull emphasis away
    lr = min(0.4, 0.2 * weight)
    for fld, vecs in _embeddings_by_field(store, cfg, liked_chunks).items():
        profile.update_vector(vecs, field=fld, lr=lr)             # toward, in that field's subspace
    for fld, vecs in _embeddings_by_field(store, cfg, disliked_chunks).items():
        profile.update_vector(vecs, field=fld, lr=lr, away=True)  # away, in that field's subspace
    profile.n_signals += 1
    save_persona(profile)
    home().mkdir(parents=True, exist_ok=True)
    with _log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "query": query,
                            "liked_projects": liked_projects, "liked_chunks": liked_chunks,
                            "disliked_projects": disliked_projects, "disliked_chunks": disliked_chunks,
                            "fields_up": up, "fields_down": down, "weight": weight}) + "\n")
    return profile
