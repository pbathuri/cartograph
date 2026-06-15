"""The vision pipeline: turn one captured frame into (maybe) one graph chunk, holistically wired.

Order is chosen for privacy + efficiency:
  1. sensitive-window denylist  -> never even OCR banking/password/auth screens
  2. image novelty gate         -> skip OCR entirely for near-duplicate screens (the similarity cache)
  3. OCR                         -> text only for genuinely-new screens
  4. min-text + text novelty     -> drop empty/again-duplicate content
  5. redact                      -> strip secrets/PII before anything is stored
  6. classify (router + intent)  -> reuse the SAME trained models the rest of Cartograph uses
  7. ingest as a graph chunk      -> flows into FTS/semantic retrieval, persona, brief automatically
  8. (opt) nudge the persona      -> field weights drift toward what you actually spend time on

Every step returns a small record describing what happened (or why it was skipped) — fully inspectable.
"""
from __future__ import annotations

import datetime as _dt
import re as _re
from dataclasses import dataclass, field as _field
from typing import Any, Callable

from ..config import Config
from ..storage import Store
from .dedup import NoveltyGate
from .redact import DEFAULT_DENYLIST, is_sensitive_window, redact

PROJECT_NAME = "screen-activity"
PROJECT_FIELD = "screen-activity"
_HEADER_RE = _re.compile(r"\[screen field=(?P<field>\S+) intent=(?P<intent>\S+) app=(?P<app>.*)\]$")


@dataclass
class VisionConfig:
    interval_sec: float = 60.0
    denylist: list[str] = _field(default_factory=lambda: list(DEFAULT_DENYLIST))
    min_chars: int = 40                 # ignore frames with almost no text
    img_threshold: int = 6              # aHash Hamming <= this => duplicate image (skip OCR)
    text_threshold: float = 0.92        # difflib ratio >= this => duplicate text
    redact: bool = True
    learn_persona: bool = True          # let observed activity gently shape persona field weights
    max_chars: int = 4000               # cap stored text per frame


def _today_path() -> str:
    return "screen://" + _dt.date.today().isoformat()


def _ingest(store: Store, text: str, field: str, intent: str, app: str) -> int:
    """Append one screen chunk under the daily 'file' of the screen-activity project. Returns chunk count."""
    pid = store.upsert_project(PROJECT_NAME, "(live screen capture)", PROJECT_FIELD)
    fid = store.ensure_file(pid, _today_path(), ext="screen")
    header = f"[screen field={field} intent={intent} app={app}]"
    return store.add_chunks(fid, [header + "\n" + text])


def _nudge_persona(persona, field: str) -> None:
    """Weak signal: you spent a tick in this field -> drift its weight up a little, decay the rest.
    Much gentler than explicit feedback; it makes the persona track 'what I'm doing now' over time."""
    if not field or field in ("general", PROJECT_FIELD):
        return
    fw = persona.field_weights
    for k in list(fw):
        fw[k] *= 0.999
    fw[field] = min(1.0, fw.get(field, 0.0) + 0.01)


def process_frame(frame, store: Store, cfg: Config, vcfg: VisionConfig, gate: NoveltyGate,
                  ocr: Callable[[Any], str], *, persona=None, apply: bool = True) -> dict:
    """Process one Frame. `apply=False` is a dry run: everything except storing/learning still runs, so
    you can preview exactly what WOULD be captured (and what gets redacted) before turning it on."""
    if frame is None:
        return {"action": "skip", "reason": "no_frame"}
    app = frame.app_title or ""
    if is_sensitive_window(app, vcfg.denylist):
        return {"action": "skip", "reason": "sensitive_window", "app": app}
    if gate.image_is_dup(frame.image):
        return {"action": "skip", "reason": "duplicate_image", "app": app}   # cache hit: no OCR cost

    text = (ocr(frame.image) or "").strip()
    if len(text) < vcfg.min_chars:
        gate.update(frame.image, text)
        return {"action": "skip", "reason": "too_little_text", "chars": len(text), "app": app}
    if gate.text_is_dup(text):
        gate.update(frame.image, text)
        return {"action": "skip", "reason": "duplicate_text", "app": app}

    clean, nred = redact(text) if vcfg.redact else (text, 0)
    clean = clean[: vcfg.max_chars]

    from ..intent import classify
    from ..router import infer_field
    field = infer_field(clean, cfg)
    intent = classify(clean)["intent"]
    gate.update(frame.image, text)

    rec = {"action": "store" if apply else "preview", "app": app, "field": field, "intent": intent,
           "chars": len(clean), "redacted": nred, "preview": clean[:160]}
    if apply:
        _ingest(store, clean, field, intent, app)
        if vcfg.learn_persona and persona is not None:
            from ..persona.profile import save_persona
            _nudge_persona(persona, field)
            save_persona(persona)
    return rec


def recent_activity(store: Store, *, limit: int = 3) -> list[dict]:
    """Latest screen-activity chunks, parsed back into {app, field, intent, snippet} for live context."""
    pid = store.project_id_by_name(PROJECT_NAME)
    if pid is None:
        return []
    out = []
    for ch in store.recent_chunks(pid, limit=limit):
        txt = ch.get("chunk_text") or ""
        meta, _, body = txt.partition("\n")
        m = _HEADER_RE.match(meta)                         # app may contain spaces -> regex, not split
        info = {"app": (m.group("app").strip() if m else ""),
                "field": (m.group("field") if m else ""),
                "intent": (m.group("intent") if m else ""),
                "snippet": body[:160]}
        out.append(info)
    return out
