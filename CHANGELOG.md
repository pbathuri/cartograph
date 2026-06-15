# Changelog

All notable changes to Cartograph (`cartograph__v1`) are documented here. Versioning is semantic.

## [1.0.0] — 2026-06-15
First public release. A local-first personal cognitive graph you point at your own folders and plug into
your AI agents.

### Core
- **Graph + hybrid retrieval** — local SQLite + FTS5; Reciprocal-Rank-Fusion of semantic (optional) and
  keyword search. Works with zero heavy ML; semantic turns on automatically once you `carto index`.
- **MCP server** (`carto mcp-server`) — read-only stdio; plug your graph into Claude Code / Cursor.
- **Visual app** (`carto viz`, `carto studio`) — explore the graph; the Studio lets non-technical users
  edit the processing workflow (add/connect/import nodes) with the essential core locked behind disclaimers.

### Personalization (the "in-between" layer, all learned from *your* data)
- Field router, prompt-intent classifier, per-field preference subspaces, learned α.
- **Learned reranker** trained from your feedback log (pure NumPy).
- **Query-contextual affinity** — preference conditioned on the *kind* of query (validated to generalize
  to unseen paraphrases; see `docs/SELF_AUDIT.md`).
- **Learned/clustered fields** (`carto fields`) — emergent fields for any vocabulary, so field features
  work for non-dev users whose corpus would otherwise collapse to `general`.
- Relevance-gated, top-hit-protected steering so personalization **does no harm** to clear queries.

### Real-time vision (optional, `cartograph__v1[vision]`)
- `carto watch` — periodic screen capture → novelty/similarity cache → OCR → redact → classify → graph.
  Local-only, dry-run by default, sensitive-window denylist, secret/PII redaction, optional encryption.

### Security
- Local server hardened: loopback-only bind, strict CORS allowlist (no `*`), DNS-rebinding Host check,
  token-gated mutations, optional at-rest encryption (`cartograph__v1[secure]`). See `SECURITY.md`.

### Validation
- End-to-end trials for a technical (computer engineering) and non-technical (finance consultant) user,
  plus a self-audit with held-out generalization + ablation. See `docs/V1_TRIAL.md`,
  `docs/AUDIT_DUAL_USERS.md`, `docs/SELF_AUDIT.md`.

## [0.2.0]
- Initial PyPI publish (Trusted Publishing via GitHub OIDC). Graph + hybrid retrieval + MCP + viz.
