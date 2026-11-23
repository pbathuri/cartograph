# Changelog

All notable changes to Cartograph (`cartograph__v1`) are documented here. Versioning is semantic.

## [Unreleased] — 2.0 series

### Index you can trust on a big corpus
- **Resumable builds.** A killed `carto index` continues from the rows it already wrote instead of
  starting over (it used to wipe a partial build). Partial builds record their chunk ids; one left by an
  older version is resumed only when the graph hasn't been written since it began.
- **Your work first.** Chunks are embedded in the order of your configured roots, smaller projects first
  within a root, so your own folders become searchable hours before bulk reference clones.
- **Searchable while building.** Retrieval uses the partial index (unfinished rows are zero vectors and
  never outrank real matches) instead of silently falling back to keyword-only.
- **`carto index --background`** runs detached from the terminal or agent session that started it;
  **`carto index --status [--watch]`** shows a live progress panel (count, rate, ETA, resume hint).
- An `index/` folder missing `chunk_ids.npy` is no longer treated as a usable index.
- fp16 inference on CUDA/MPS (~15% faster on Apple silicon, identical vectors at float16 storage).

### Change detection
- Files are hashed in full (v1 hashed only the first 1 MB, so edits past it went unnoticed). Files up to
  1 MB keep their v1 hash, so upgrading re-chunks only the 1–2 MB files that could have been stale.
- Size + mtime are stored per file; unchanged files are skipped without being read.
- Existing graphs are migrated in place (`ALTER TABLE ... ADD COLUMN`, no rebuild).

### MCP
- Real JSON Schemas: array parameters (`record_use.helped`) are `array` of `string`, integers are integers.
- Tool `annotations` (`readOnlyHint`, `destructiveHint`, ...) distinguish retrieval from tools that write
  local state; `carto mcp-server --read-only` hides the writers.
- Tool failures come back as `isError` results (spec behaviour) rather than JSON-RPC errors; protocol
  version negotiation (2024-11-05 / 2025-03-26 / 2025-06-18); `ping`.
- A long-running server that outlives an upgrade heals stale imports instead of failing every call.
- Clear message when no graph exists yet. `python -m cartograph` works.

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
