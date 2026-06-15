"""The novelty / similarity cache — the reasoned 'model in between' for a background CV loop.

A screen barely changes between most ticks. Re-OCRing and re-embedding identical frames wastes compute
and floods the graph with duplicates. So before any expensive work we ask: *is this frame new enough?*

Two cheap signals, no heavy model:
  * image: a 64-bit average-hash (aHash) + Hamming distance — robust to tiny pixel changes, ~free.
    The image check gates OCR itself, so duplicate screens cost almost nothing.
  * text: difflib ratio on the OCR'd text — dependency-free; catches "same content, moved cursor".

A frame is a duplicate when it is similar on the available signal(s). This is literally similarity
caching: the gate is the cache, and a hit means "skip, we already know this screen."
"""
from __future__ import annotations

import difflib


def ahash(image) -> int | None:
    """64-bit average hash of a PIL image. Returns None if PIL/image unavailable (text-only fallback)."""
    if image is None:
        return None
    try:
        from PIL import Image  # noqa: F401
        g = image.convert("L").resize((8, 8))
        px = list(g.getdata())
        avg = sum(px) / len(px)
        bits = 0
        for i, p in enumerate(px):
            if p >= avg:
                bits |= (1 << i)
        return bits
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class NoveltyGate:
    """Holds the last accepted frame's signals; decides whether a new frame is worth processing."""

    def __init__(self, img_threshold: int = 6, text_threshold: float = 0.92) -> None:
        self.img_threshold = img_threshold        # Hamming <= this => images "the same"
        self.text_threshold = text_threshold      # difflib ratio >= this => texts "the same"
        self._last_hash: int | None = None
        self._last_text: str = ""

    def image_is_dup(self, image) -> bool:
        h = ahash(image)
        if h is None or self._last_hash is None:
            return False
        return hamming(h, self._last_hash) <= self.img_threshold

    def text_is_dup(self, text: str) -> bool:
        if not self._last_text:
            return False
        return difflib.SequenceMatcher(None, self._last_text, text or "").ratio() >= self.text_threshold

    def update(self, image, text: str) -> None:
        h = ahash(image)
        if h is not None:
            self._last_hash = h
        self._last_text = text or ""
