"""OCR — pluggable text extraction. The 'vision model' is the OCR engine itself; we keep it swappable.

Default backend is Tesseract via pytesseract (needs the Tesseract binary on PATH). Optional dep
(`cartograph[vision]`). Tests inject a plain callable, so no engine is required to exercise the pipeline.
"""
from __future__ import annotations

from typing import Callable


def tesseract_ocr(image) -> str:
    if image is None:
        return ""
    try:
        import pytesseract
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def default_ocr() -> Callable[[object], str] | None:
    """Return an OCR callable if a backend is available, else None."""
    try:
        import pytesseract  # noqa: F401
        return tesseract_ocr
    except Exception:
        return None
