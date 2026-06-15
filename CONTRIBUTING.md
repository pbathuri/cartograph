# Contributing to Cartograph

Thanks for your interest! Cartograph is a local-first personal cognitive graph for AI work. Contributions
are welcome — bug reports, docs, tests, and features.

## Development setup
```bash
git clone https://github.com/pbathuri/cartograph
cd cartograph
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # core + pytest + numpy + ruff (no ~2GB model)
pip install -e ".[full]"         # optional: semantic/ml/vision/secure to run every test locally
```

## Before opening a PR
```bash
ruff check cartograph tests       # lint (CI enforces this)
python -m pytest tests -q         # tests — semantic-only tests auto-skip without [semantic]
python -m build && twine check dist/*   # package still builds clean
```
CI runs the same on Linux/macOS/Windows for Python 3.10 and 3.12.

## Guidelines
- **Local-first & private by design.** Never add code that uploads user data or phones home. No personal
  data, secrets, or real corpora in the repo.
- **Degrade gracefully.** The core (keyword + graph + viz + MCP) must work with zero heavy ML; semantic /
  vision / encryption are optional extras that no-op cleanly when their deps are absent.
- **Match the surrounding style.** Compact, commented where intent isn't obvious; pure-NumPy for the small
  learned models (no sklearn). Keep `ruff check` clean.
- **Test what you change**, and be honest in docs about what's measured vs aspirational (see `docs/`).

## Reporting bugs
Open an issue with your OS, Python version, `carto doctor` output, and steps to reproduce. For security
issues, see [SECURITY.md](SECURITY.md) — please use a private advisory, not a public issue.
