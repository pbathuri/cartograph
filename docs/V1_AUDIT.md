# Cartograph V1 — readiness audit

This records the audit behind declaring **v1.0.0**. The bar: the core model is stable and unchanged,
the surfaces are hardened, privacy is enforced by design, and everything is covered by tests.

## 1. Core model — stable, NOT changed for V1
The retrieval/persona/learning core was left intact. V1 adds only *surfaces* and *guardrails* around it:
- Hybrid retrieval (RRF of semantic + FTS), per-field preference subspaces, learned α, learned reranker,
  field router, intent classifier — all unchanged from the 0.x line.
- New in the 1.0 cycle: the real-time **vision** data source, the **security** layer, the **workflow**
  overlay + **Studio** app, and the **insights** stats — all additive and isolated.

## 2. Security audit (see SECURITY.md for the threat model)
Reviewed every new surface for injection, authz, crypto, deserialization, and data exposure.

| Area | Result |
|---|---|
| Local server CORS | **Fixed a High** introduced during hardening: an unanchored `startswith` loopback check allowed origin spoofing (`http://127.0.0.1.attacker.com`). Now an anchored regex; exact allowlist otherwise. Regression-tested. |
| CSRF on mutations | All state-changing endpoints require a per-workspace token (`secrets` + constant-time `hmac.compare_digest`). |
| DNS rebinding | Host-header loopback check; non-local → 403. |
| Network exposure | Binds `127.0.0.1` only. |
| Deserialization | No `pickle`/`yaml.load`/`eval`; `np.load` uses default `allow_pickle=False`; config via `yaml.safe_load`; overlays via `json.loads`. |
| Path traversal | Static assets served from fixed paths; no user-supplied file paths in the server. |
| XSS (Studio) | User-controlled strings escaped (`esc()`); only server-generated ids interpolated into attributes. |
| Crypto | Optional Fernet (AES-128-CBC + HMAC) via `cryptography`; key/token written owner-only. |
| MCP server | Read-only stdio; no network port. |

## 3. Privacy
- Local-first; nothing leaves the device via the package. No personal data in the repo (audited).
- Vision: sensitive-window denylist (never OCR'd), secret/PII redaction **before** storage, opt-in
  encryption-at-rest, dry-run default, pause kill-switch.
- A user-connected cloud LLM is the user's explicit choice, shown with a privacy disclaimer first.

## 4. The Studio's locked core (integrity of the essential path)
The essential spine — **redact · router · persona · retrieval · brief** — is locked. Delete/disconnect
attempts return a disclaimer explaining what would break (e.g. removing `redact` risks leaking secrets),
never a silent break. Optional stages (vision, intent, reranker) can be disabled, not destroyed. User
edits live in a `workflow.json` overlay; the cognitive-graph DB is never destructively modified.

## 5. Tests
**55 passing**, including: security (token, anchored CORS allowlist + spoof regressions, host check,
encryption round-trip), workflow (lock enforcement, optional-disable, model import, edge locks, overlay
persistence), a live HTTP server suite (token gating, locked→423, origin not echoed), plus the existing
retrieval/persona/router/intent/reranker/vision coverage. Wheel builds self-contained; `twine check`
passes; modules verified present in the wheel.

## Verdict
**V1 criteria met.** Core stable, surfaces hardened (one self-introduced High found and fixed),
privacy enforced, locked-core integrity guaranteed, full green suite. Tagged-release is the maintainer's
gated action: `git tag v1.0.0 && git push --tags`.
