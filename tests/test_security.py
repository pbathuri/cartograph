"""V1 security: token CSRF defense, CORS allowlist, DNS-rebinding host check, optional encryption."""
from cartograph import security


def test_token_is_stable_and_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))
    t = security.api_token()
    assert t and security.api_token() == t            # persisted, stable
    assert security.token_ok(t)
    assert not security.token_ok("nope") and not security.token_ok(None)


def test_cors_allowlist_never_wildcards(monkeypatch, tmp_path):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))
    assert security.cors_origin("https://chatgpt.com") == "https://chatgpt.com"
    assert security.cors_origin("http://127.0.0.1:8787") == "http://127.0.0.1:8787"
    assert security.cors_origin("http://localhost") == "http://localhost"
    assert security.cors_origin("https://evil.example") is None    # not echoed
    assert security.cors_origin(None) is None
    # origin-spoof regression: an unanchored prefix match would have allowed these — they MUST be refused
    assert security.cors_origin("http://127.0.0.1.attacker.com") is None
    assert security.cors_origin("http://localhost.attacker.com") is None
    assert security.cors_origin("http://127.0.0.1:8787.evil.com") is None


def test_host_check_blocks_rebinding():
    assert security.host_is_local("127.0.0.1:8787")
    assert security.host_is_local("localhost:8787")
    assert not security.host_is_local("attacker.com:8787")


def test_encryption_roundtrip_or_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPH_HOME", str(tmp_path))
    enc = security.encrypt_text("secret password hunter2")
    dec = security.decrypt_text(enc)
    if security.encryption_available():
        assert enc.startswith("ENC1:") and "hunter2" not in enc and dec == "secret password hunter2"
    else:
        assert enc == "secret password hunter2"        # graceful no-op without cryptography
