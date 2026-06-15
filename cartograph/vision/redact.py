"""Privacy core: redact secrets/PII from OCR'd text, and skip windows that should never be captured.

This runs BEFORE anything is stored, so sensitive data never enters the graph. Conservative by design —
better to over-redact than to leak. Both the denylist and patterns are extendable via VisionConfig.
"""
from __future__ import annotations

import re

# Window/app titles we refuse to OCR at all (substring match, case-insensitive). Banking, password
# managers, auth screens, payment flows, private/incognito browsing.
DEFAULT_DENYLIST = [
    "password", "1password", "bitwarden", "lastpass", "keepass", "dashlane", "keychain",
    "bank", "banking", "chase", "wells fargo", "paypal", "venmo", "stripe", "coinbase", "wallet",
    "sign in", "log in", "login", "authenticator", "two-factor", "2fa", "recovery code",
    "private browsing", "incognito", "credit card", "checkout", "billing",
]

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),                 # card-like digit runs
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"), "[api-key]"),             # OpenAI-style
    (re.compile(r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{12,}\b"), "[token]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"), "[jwt]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[aws-key]"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "[hex-secret]"),             # long hex (keys/hashes)
    (re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*\S+"), r"[credential]"),
    (re.compile(r"\b(?:\+?\d[ -]?){10,15}\b"), "[phone]"),
]


def is_sensitive_window(title: str | None, denylist: list[str] | None = None) -> bool:
    t = (title or "").lower()
    return any(term in t for term in (denylist or DEFAULT_DENYLIST))


def redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, num_redactions). Order matters: structured secrets before generic phone."""
    n = 0
    for pat, repl in _PATTERNS:
        text, k = pat.subn(repl, text)
        n += k
    return text, n
