# Security & privacy

Cartograph is **local-first**: your graph, persona, and any screen captures live only in your workspace
(`~/.cartograph` or `$CARTOGRAPH_HOME`) and are never uploaded anywhere by the package. This document is
the V1 threat model and the hardening that ships.

## Threat model

The package has no remote server, so the realistic adversary is **a web page you visit while a local
Cartograph server is running** (`carto serve` / `carto viz` / `carto studio`). Browsers let any page send
requests to `http://127.0.0.1:<port>`. Two classic attacks follow, and both are defended:

| Threat | Defense (shipped) |
|---|---|
| **Drive-by read / CSRF** — a malicious site fetches `/api/graph` or `/api/personalize` to exfiltrate your data | Strict **CORS allowlist** (known GenAI origins + loopback only; the wildcard `*` is gone, so other origins can't read responses). Every **state-changing** endpoint requires a per-workspace **API token** the page cannot guess. |
| **DNS rebinding** — a site rebinds its hostname to 127.0.0.1 to bypass same-origin | **Host-header check** on every request: non-loopback hosts are refused (`403`). |
| **Remote network exposure** | Server **binds `127.0.0.1`** only — never `0.0.0.0`. |
| **Secrets/PII captured by the vision layer** | Sensitive windows (banking/passwords/auth) are **never OCR'd**; secrets + PII are **redacted before storage**; screen text can be **encrypted at rest** (`encrypt_at_rest`, `cartograph[secure]`). |
| **Token/key theft from disk** | Token + key files are written with **owner-only permissions** (POSIX `chmod 600`; on Windows they inherit the user-profile ACL — keep your OS account secured). |

The **MCP server** (`carto mcp-server`) is **stdio + read-only**: it cannot mutate your graph or persona
state-changing surfaces, and it never opens a network port.

## What is stored, and where

- Graph (`graph.sqlite`), persona (`persona.json`, `persona_vecs.npz`), feedback log (`feedback.jsonl`),
  learned models (`field_centroids.npz`, `reranker.npz`), workflow overlay (`workflow.json`),
  API token (`.api_token`), optional encryption key (`.enc_key`). All under your workspace, all local.
- **No personal data ships in the package**, and nothing is transmitted off-device by Cartograph itself.
  If *you* wire in a cloud LLM through the Studio, that is your explicit choice and is shown with a
  privacy disclaimer first.

## Encryption at rest (optional, opt-in)

`pip install 'cartograph__v1[secure]'` enables Fernet (AES-128-CBC + HMAC) via the `cryptography`
library. With `VisionConfig.encrypt_at_rest=True`, captured screen text is encrypted before it touches
the database (metadata like field/intent stays clear for classification). Trade-off: encrypted screen
text is **not** keyword-searchable (FTS indexes ciphertext). Default is **off** so retrieval works; turn
it on for privacy-max mode. Run `carto secure` to see your posture.

## What is intentionally NOT claimed

- Cartograph does **not** hook into closed models' (ChatGPT/Gemini) internal generation — no such hook
  exists. It enriches the *context injected before* a prompt. See `docs/PERSONA.md`.
- App-level encryption protects data at rest from casual disk access; it is **not** a defense against
  malware running as your user, which can read your decrypted workspace like you can. Use full-disk
  encryption (BitLocker/FileVault/LUKS) for that layer.

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository (Security → Report a vulnerability)
rather than a public issue. We aim to acknowledge within a few days.
