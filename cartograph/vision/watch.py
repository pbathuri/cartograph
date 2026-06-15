"""The capture loop. Runs process_frame every `interval_sec`. Injectable capturer/ocr/sleep so it is
fully testable (finite `iterations`, fake clock). Local-only; honors a pause file for a kill-switch."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ..config import Config, home
from ..storage import Store
from .dedup import NoveltyGate
from .pipeline import VisionConfig, process_frame


def pause_path() -> Path:
    return home() / "vision.paused"


def watch(store: Store, cfg: Config, vcfg: VisionConfig, capturer, ocr: Callable[[Any], str], *,
          iterations: int | None = None, apply: bool = True, persona=None,
          on_record: Callable[[dict], None] | None = None,
          sleep: Callable[[float], None] = time.sleep) -> dict:
    """Loop until `iterations` is reached (None = forever). Returns a summary of actions taken.
    Create the file `~/.cartograph/vision.paused` to pause without killing the process."""
    gate = NoveltyGate(vcfg.img_threshold, vcfg.text_threshold)
    counts: dict[str, int] = {}
    i = 0
    while iterations is None or i < iterations:
        if pause_path().exists():
            rec = {"action": "skip", "reason": "paused"}
        else:
            rec = process_frame(capturer.capture(), store, cfg, vcfg, gate, ocr,
                                persona=persona, apply=apply)
        key = f"{rec['action']}:{rec.get('reason', rec.get('field', ''))}"
        counts[key] = counts.get(key, 0) + 1
        if on_record:
            on_record(rec)
        i += 1
        if iterations is None or i < iterations:
            sleep(vcfg.interval_sec)
    return {"ticks": i, "counts": counts}
