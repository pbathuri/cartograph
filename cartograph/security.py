"""Security primitives for the local surfaces — added for V1 WITHOUT touching the core model.

Threat model (local-first app): the dangerous attacker is not a remote server (we bind to loopback) but
*any web page the user visits while a local server is running*. A page can issue requests to
http://127.0.0.1:<port>, so two classic attacks apply:
  * CSRF / drive-by reads — a malicious site fetches /api/graph or /api/personalize and exfiltrates the
    user's persona / graph / OCR'd screen text.  -> Defense: a per-workspace API token + a strict CORS
    allowlist (reflect only known origins), never `*`.
  * DNS-rebinding — a site rebinds its hostname to 127.0.0.1 to bypass same-origin.  -> Defense: a Host
    header check (must be loopback) on every request.

Plus at-rest protection for the most sensitive new data (screen captures): optional Fernet encryption.
All of this is additive and isolated; the retrieval/persona/vision core is unchanged.
"""
from __future__ import annotations

import hmac
import os
import secrets
import stat
from pathlib import Path

from .config import home

# Origins allowed to talk to the local API from a browser. The web-GenAI userscript runs on these;
# everything else is refused (no `*`). Loopback is allowed for the studio app itself.
ALLOWED_ORIGINS = {
    "https://chatgpt.com", "https://chat.openai.com", "https://gemini.google.com",
    "https://claude.ai", "https://www.bing.com", "https://copilot.microsoft.com",
}


def _secure_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    harden_perms(path)


def harden_perms(path: Path) -> None:
    """Best-effort owner-only perms. On POSIX this is chmod 600; on Windows the file inherits the user
    profile ACL (documented in SECURITY.md) and we at least clear group/other where supported."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


# ---- API token (CSRF defense) ----
def _token_path() -> Path:
    return home() / ".api_token"


def api_token() -> str:
    """Read or create the per-workspace API token. Mutating endpoints require it; it lives only on disk
    in the user's workspace, so a remote page cannot guess it."""
    p = _token_path()
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    tok = secrets.token_urlsafe(32)
    _secure_write(p, tok.encode("utf-8"))
    return tok


def token_ok(provided: str | None) -> bool:
    if not provided:
        return False
    return hmac.compare_digest(provided, api_token())   # constant-time


# ---- request origin / host checks ----
def cors_origin(origin: str | None) -> str | None:
    """Return the origin to echo in Access-Control-Allow-Origin, or None to send no CORS header."""
    if not origin:
        return None
    o = origin.rstrip("/")
    if o in ALLOWED_ORIGINS or o.startswith("http://127.0.0.1") or o.startswith("http://localhost"):
        return o
    return None


def host_is_local(host_header: str | None) -> bool:
    """Reject DNS-rebinding: the Host must resolve to loopback by name."""
    if not host_header:
        return True                                      # no Host (older clients) — binding already loopback
    h = host_header.split(":")[0].strip().lower()
    return h in ("127.0.0.1", "localhost", "::1", "[::1]")


# ---- optional encryption-at-rest (screen captures) ----
ENC_PREFIX = "ENC1:"


def encryption_available() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except Exception:
        return False


def _key_path() -> Path:
    return home() / ".enc_key"


def _get_key():
    from cryptography.fernet import Fernet
    p = _key_path()
    if p.exists():
        return Fernet(p.read_bytes().strip())
    key = Fernet.generate_key()
    _secure_write(p, key)
    return Fernet(key)


def encrypt_text(text: str) -> str:
    """Return ENC1:<token>. No-op (returns text unchanged) if cryptography isn't installed."""
    if not encryption_available():
        return text
    return ENC_PREFIX + _get_key().encrypt(text.encode("utf-8")).decode("ascii")


def decrypt_text(text: str) -> str:
    """Inverse of encrypt_text; passes through anything without the marker."""
    if not text or not text.startswith(ENC_PREFIX):
        return text
    if not encryption_available():
        return "[encrypted — install cartograph__v1[secure] to read]"
    try:
        return _get_key().decrypt(text[len(ENC_PREFIX):].encode("ascii")).decode("utf-8")
    except Exception:
        return "[encrypted — key mismatch]"
