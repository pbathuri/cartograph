"""Screen capture — pluggable so the pipeline is testable without a real screen.

Real backend uses `mss` (fast, cross-platform) for the image and best-effort active-window title for the
denylist check. Both are optional deps (`cartograph[vision]`); absence degrades gracefully. Tests inject
a FakeCapturer, so nothing here needs a display.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Frame:
    image: Any          # a PIL.Image or None (None => text-only / fake)
    app_title: str      # active window/app title, for the sensitive-window denylist
    ts: float           # epoch seconds


def active_window_title() -> str:
    """Best-effort foreground window title. Returns '' if unavailable (never raises)."""
    try:                                                   # Windows
        import ctypes
        u = ctypes.windll.user32
        h = u.GetForegroundWindow()
        n = u.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, buf, n + 1)
        return buf.value or ""
    except Exception:
        pass
    try:                                                   # cross-platform fallback
        import pygetwindow as gw
        w = gw.getActiveWindow()
        return getattr(w, "title", "") or ""
    except Exception:
        return ""


class MSSCapturer:
    """Captures the primary monitor via mss into a PIL image."""

    def __init__(self) -> None:
        import mss                                          # raises if not installed -> caller handles
        from PIL import Image  # noqa: F401
        self._sct = mss.mss()

    def capture(self) -> Frame | None:
        try:
            from PIL import Image
            mon = self._sct.monitors[1]
            raw = self._sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            return Frame(image=img, app_title=active_window_title(), ts=time.time())
        except Exception:
            return None


class FakeCapturer:
    """Cycles through canned (image, title) frames — for tests and dry demos."""

    def __init__(self, frames: list[tuple[Any, str]]) -> None:
        self._frames = frames
        self._i = 0

    def capture(self) -> Frame | None:
        if not self._frames:
            return None
        img, title = self._frames[self._i % len(self._frames)]
        self._i += 1
        return Frame(image=img, app_title=title, ts=time.time())


def default_capturer():
    """Real capturer if deps are present, else None (CLI then tells the user to install [vision])."""
    try:
        return MSSCapturer()
    except Exception:
        return None
