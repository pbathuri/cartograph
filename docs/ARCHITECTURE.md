# Architecture

Cartograph is a small, local-first pipeline. Every box below is one module you can read in a sitting.

```
your folders ──▶ ingest.py ──▶ storage.py (SQLite + FTS5) ──▶ retrieve.py (hybrid RRF)
                    │                  │                            │
              field inference     graph_sample()                embed.py (optional vectors)
                                       │                            │
                                  viz/app.py (browser)        elite/ (catalog·playbooks·dod·frontier·elevate)
                                       │                            │
                                       └──────── cli.py · mcp_server.py ───────┘
```

## Modules
| File | Responsibility |
|---|---|
| `config.py` | workspace paths (`~/.cartograph` or `CARTOGRAPH_HOME`) + YAML config |
| `storage.py` | the graph: `projects · files · chunks · skills · edges` + FTS5; read-only mode |
| `ingest.py` | walk folders → chunk → store (incremental, hash-based); field inference; relation edges |
| `embed.py` | optional semantic index (sentence-transformers; GPU-aware; numpy memmap) |
| `retrieve.py` | **hybrid retrieval** — Reciprocal Rank Fusion of semantic + FTS; degrades to FTS-only |
| `elite/` | field-agnostic frontier layer: reference catalog, playbooks, Definition-of-Done, coverage, `elevate` |
| `mcp_server.py` | stdio JSON-RPC MCP server for Claude Code / Cursor / any agent |
| `viz/` | zero-dependency local web visualizer (stdlib http.server + vis-network) |
| `cli.py` | the `carto` command surface |

## Design choices (and why)
- **SQLite + FTS5, single file.** Zero servers, portable, scales to millions of chunks. The graph is
  one file you own and can back up or move.
- **Hybrid retrieval by default.** Semantic catches paraphrases; keyword catches exact/rare tokens.
  RRF fuses them with no score normalization and is *never worse* than either alone.
- **Heavy ML is opt-in.** Core install has no torch/transformers — instant, works anywhere. Semantic
  and training are extras you add when you want them.
- **Local-first, privacy by default.** Nothing leaves your machine; data dirs are git-ignored.
- **Field-agnostic elite layer.** The catalog/playbooks/DoD cover many fields and are one file each to
  extend — Cartograph elevates *your* field, not a fixed one.

## Honest limits
- Build grades (`carto review`) are filename + content-marker heuristics: a marker shows a practice was
  *referenced*, not that it is *correct*. Confirm with eval/human review.
- Brute-force semantic search is great to ~few-million chunks; past that, see [SCALING.md](SCALING.md).
