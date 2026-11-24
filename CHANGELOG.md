# Changelog

All notable changes to Cartograph (`cartograph__v1`) are documented here. Versioning is semantic.

## [2.0.3] — 2026-11-23

Found by auditing a real 7.1M-chunk, 604k-file graph with a 30-question workflow eval. The eval
ships as `scripts/workflow_eval.py` (with an example question file) so anyone can score Cartograph on
their own files against ripgrep; the README has the results.

- **Re-ingest removes files that are gone.** Ingest never deleted a file row: deleted files kept being
  cited, and when a folder ingested as one project was later split into its repos, every file stayed in
  both. On the audited graph that was 233,386 files and 2.3M chunks (a third of the index) stored twice,
  so most results came back in pairs. Files deleted from disk, newly ignored, or moved into a sub-project
  are now pruned (`removed N file(s)` in the ingest summary). Unmounted drives are still left alone.
- **Files at the top of a root folder are ingested.** When a root held several projects, its own loose
  files (a manuscript, meeting notes) were skipped. On the audited machine that was 979 files.
- **Keyword search is about 10x faster on large graphs.** Question words (how, do, with, where...) no longer
  become search terms, and BM25 ranks inside the FTS table before joining. On the 7.1M-chunk graph median
  keyword latency fell from 1.8 s to 0.16 s and hybrid `retrieve_context` from 1.9 s to 0.4 s; SciFact nDCG@10 rose (keyword 0.655 -> 0.658,
  hybrid 0.709 -> 0.714).
- **Naming a project finds its code.** "Where is causal self-attention defined in nanoGPT" now searches
  inside nanoGPT too (with double weight in the fusion), because a repo's own files rarely mention its name.
- **The same passage is never returned twice.** Identical text from a second checkout or copy of a file
  collapses to its best-ranked citation.
- **Mixed answers skip off-topic session logs.** Auto-captured session notes are long enough to match any
  single word, so they crowded `mode: mix` answers. They now need half the query's keywords; decisions,
  rejections, preferences and facts still need only one.
- **No 10-second first call.** `carto mcp-server` loads the embedding model and index on a background
  thread at start (`CARTOGRAPH_MCP_WARM=0` turns this off).
- `CARTOGRAPH_EMBED_CHARS` sets how much of each chunk is embedded (default 600). 1200 scores SciFact
  hybrid nDCG@10 0.723 instead of 0.714, at about twice the embedding time.
- **Index builds can no longer destroy each other.** A second builder (e.g. the nightly `carto improve`
  starting while `carto index` runs) used to delete the first one's folder and start over, while the
  first kept writing into the deleted file. Builds now take a single-builder lock; `improve` skips its
  index step while a build is running, and `carto index` says so.
- **Resume survives graph changes.** If chunks were added or changed since an unfinished build began, its
  finished rows are carried over by chunk id instead of thrown away; only new chunks are embedded.
- `improve` logs why a step failed and writes `reports/latest.json` after every step, so a killed run
  still leaves a record.
- Apple-GPU builds use batch 128 (about 10% faster than 64 on an M-series Mac).
- **Fresh `[semantic]` installs work again.** transformers 5.17+ removed an API the default nomic-embed
  model calls (`'NomicBertModel' object has no attribute 'get_extended_attention_mask'`); the
  `semantic` and `full` extras now require `transformers<5.17`.

## [2.0.2] — 2026-11-23

- **Listed in the official MCP Registry** as `io.github.pbathuri/cartograph` (`server.json`; the release
  workflow publishes it after PyPI via GitHub OIDC). New `cartograph__v1` command alias so
  `uvx cartograph__v1 mcp-server` works. A `Dockerfile` and `glama.json` for Glama and container use.
- `carto demo` now shows what 2.0 does, in under a second on a synthetic lab: a question answered with
  citations to a Word section, a slide and a notebook cell; two projects linked because they cite the same
  paper; a decision saved to memory, briefed to the next session, and flagged stale when its file changes.
  It cleans up after itself and leaves `CARTOGRAPH_HOME` as it was.
- Memory finds the right project when an agent reports a symlinked working directory (for example `/tmp`
  on macOS). Before, those memories were saved as global.
- Hugging Face links at the end of a sentence (`huggingface.co/org/model.`) no longer keep the full stop,
  so the same model is one entity across files and projects link correctly.
- README: leads with cited answers from your own files, answers "why not CLAUDE.md or grep?", lists the
  limits, and states exactly when the network is used. It also shows the demo above the fold.
- Moley, Your Local Guide: the mascot is redrawn as a crisp SVG badge, a nerdy mole holding a brass lantern,
  named after the mole in *The Wind in the Willows*. New README banner, social card and a favicon that still
  reads at 16 px. `scripts/brand.py` rebuilds every brand asset from code, including the demo screenshot.

## [2.0.1] — 2026-11-23
- `carto connect` no longer reports Claude Code as installed on every machine (its config lives in
  `$HOME`); it now looks for `~/.claude.json`, `~/.claude/` or the `claude` command.

## [2.0.0] — 2026-11-23
Documents with citations, a graph with provenance, memory every agent shares, and an Atlas to see it.
No LLM is needed to build any of it.

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

### Documents, with citations
- **PDF, DOCX, PPTX, XLSX, EPUB and HTML are ingested.** Office/EPUB/HTML need no extra dependency;
  PDF uses `pypdf` (`pip install "cartograph__v1[docs]"`) or PyMuPDF when present, with optional OCR of
  text-less pages (`ocr: true`). Documents get their own size cap, `max_doc_mb` (default 50).
- **Every chunk carries a locator** and every retrieval result a `cite`: `src/train.py:L40-88`,
  `paper.pdf p.12`, `talk.pptx slide 3`, `doses.xlsx sheet Doses`, `thesis.docx § Methods`,
  `analysis.ipynb cell 4-6`. Chunks never straddle pages or slides.
- Chunks from graphs built before 2.0 get line citations by re-reading the source file on demand
  (skipped quietly when the file is gone, e.g. on an unplugged drive).
- HTML is indexed as text (scripts/styles dropped) instead of raw markup. More source types:
  CUDA, Lua, Dart, Haskell, Elixir, Zig, Fortran, Protobuf, GraphQL, Terraform, BibTeX, MDX, Quarto, Typst...
- Office lock files (`~$x.docx`) are skipped; `carto ingest` reports how many PDFs were skipped for lack
  of a PDF reader.

### A graph with provenance (no LLM)
- **Deterministic entities and relations**: dependencies declared in manifests (requirements, pyproject,
  package.json, Cargo.toml, go.mod, environment.yml), Python/JS/TS imports, papers by DOI and arXiv id,
  GitHub repos, Hugging Face models and datasets.
- Every fact records its **evidence chunk** (citable), **extractor**, **confidence** and **first/last seen**;
  a file edit keeps first_seen for facts it still states and drops the ones it no longer does.
- **Projects linked through shared entities**, weighted by rarity (sharing a paper beats sharing numpy),
  with the shared entities listed on the edge.
- `carto entities PROJECT`, `carto entities --find pydantic`, `carto entities --backfill` (one pass over an
  existing graph; 6.5M chunks took ~15 min), and an `entities` MCP tool.

### Memory every agent shares
- **Temporal memory** in the same SQLite file: decisions, rejected approaches, preferences, facts, todos,
  notes and session logs. Nothing is overwritten — `supersede` closes a memory's validity interval and
  points at its replacement; history stays queryable.
- **Staleness**: memories anchored to files are flagged when those files change or disappear (files on an
  unmounted drive are reported as unavailable, not missing).
- **Conflict hints**: `remember` returns similar active memories of the same project so the agent can
  supersede instead of piling up contradictions.
- **`brief`**: a session-start digest for the project at the cwd under a hard character budget.
- **Automatic capture through client hooks** — `carto hooks install` for Claude Code, Cursor, Gemini CLI
  (hooks) and Codex (`notify`). Session start injects the brief; the session's goal, edited files and
  commands become one `session` memory anchored to those files; `decision: …` / `rejected: …` / `todo: …`
  lines said in chat become memories. Existing hooks are preserved and the config is backed up first.
- MCP tools `brief`, `recall` (read-only) and `remember`, `supersede` (write; hidden by `--read-only`).
- CLI: `carto memory add | recall | supersede | forget | brief`, `carto hooks install | uninstall | status`.

### Retrieval with hard budgets
- **Modes**: `mix` (memories + graph facts + cited chunks), `chunks`, `projects`, `graph`, `memory` — on
  `carto retrieve --mode` and the `retrieve_context` MCP tool.
- **`max_tokens` is a hard cap**: text is trimmed at paragraph/sentence boundaries and the lowest-ranked
  items are dropped until the whole payload fits. Near-duplicate chunks of the same file are collapsed.
- **Fast semantic search on millions of chunks without new dependencies**: an IVF-PQ index in pure NumPy
  (`carto index --ann`, built automatically for 100k+ rows) with exact rescoring of the candidates.
  On a 6.5M-chunk graph: 7 ms p50 / 11 ms p95 versus 8–10 s for brute force, tie-aware recall@10 0.99.
  (FAISS was rejected: on macOS its OpenMP runtime collides with PyTorch's.)
- A rebuilt index is picked up by a running MCP server without a restart.
- Identifiers such as `IL-6`, `KX-417`, `CVE-2024-1234` or `torch.nn` are also matched as exact phrases, so
  their fragments appearing elsewhere no longer outrank the real thing.

### Reproducible benchmarks
- **`carto bench`** runs in a throwaway home (your graph is never touched) and prints a Markdown table:
  - **BEIR SciFact** (5,183 abstracts, 300 labelled queries): nDCG@10, recall@10/100, MRR@10, latency, for
    keyword and (`--semantic`) hybrid retrieval. Downloads from BEIR or, when that host is down, the Hugging
    Face mirror.
  - **Citations**: a generated library of PDFs, Word files, slide decks, notes and code with planted facts on
    known pages, sections, slides and lines; scores whether the top result is the right file *and* the
    citation points at the right place.
- Results are kept per day in `~/.cartograph/bench/results-<date>.json`.

### Plugs into everything
- **`carto connect`** registers the MCP server with Claude Code, Claude Desktop, Cursor, Windsurf, VS Code,
  Cline, Gemini CLI, Codex, Zed, Continue and LM Studio (Goose: printed snippet). It backs up each file,
  is idempotent, and is reversible with `--remove`. Files with comments are never rewritten; the snippet is
  printed instead. With no arguments it shows which clients are installed and connected. The server is
  launched with the installing interpreter, so GUI apps without conda or venv on PATH still find it.
- `carto agent-setup` prints the same robust config.

### The Atlas (`carto viz`)
- Rebuilt as an offline canvas map (no CDN): a **headlamp** that lights the map around the cursor, search
  that makes matching projects glow and pulse, results with click-to-copy **citation chips**, a
  **project panel** (dependencies, citations and references with evidence; linked projects and the shared
  entities behind each link; memories), and a **memory timeline** with superseded entries struck through.
- Entity hubs (papers, repos, models, datasets shared by several projects) are drawn as diamonds.
- The map payload is cached until the graph file changes: the first load on a 6.5M-chunk graph takes
  seconds, later loads 10 ms. If the port is busy, the server moves to the next free one.
- Meet Moley, the mole, on the map, in the README and on the social card.

### Faster and cleaner
- Keyword-only search no longer imports PyTorch to find out whether semantic search is installed. The first
  query without an index went from ~10 s to 30 ms, which also speeds up MCP server cold starts.
- Interpreters and tooling (`python`, `pip`, `setuptools`, `ipykernel`, ...) are no longer graph entities
  or reasons to link two projects.

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
