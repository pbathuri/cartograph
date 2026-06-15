# Publishing Cartograph to PyPI

The package is configured and builds clean (`twine check` passes). To publish so anyone can
`pip install cartograph__v1`, run these from the repo root.

## Easiest: automated Trusted Publishing (no token) — recommended
A GitHub Actions workflow (`.github/workflows/release.yml`) publishes to PyPI on every version tag,
with **no API token stored anywhere** (uses OIDC). One-time setup:
1. Create the project on PyPI (first manual upload below, or a pending publisher).
2. PyPI → **Publishing** → add a **GitHub** publisher: repo `pbathuri/cartograph`,
   workflow `release.yml`, **Environment name: leave BLANK** (the workflow declares no environment, so
   they must match — a blank-vs-set mismatch causes a `403 OIDC token not valid for project`).
3. Then forever after:
   ```bash
   # bump version in pyproject.toml + cartograph/__init__.py, commit, then:
   git tag v0.2.0 && git push --tags
   ```
   The workflow builds, `twine check`s, and publishes. Done.

---

## Manual path (API token)

## 0. One-time: get a PyPI API token
Create a token at https://pypi.org/manage/account/token/ (scope it to this project after the first
upload). Never commit it. Use it as the password with username `__token__`, or put it in `~/.pypirc`.

## 1. Build the artifacts
```bash
python -m pip install --upgrade build twine
rm -rf dist build *.egg-info          # Windows: rmdir /s /q dist build
python -m build                       # -> dist/cartograph-0.1.0-py3-none-any.whl + .tar.gz
python -m twine check dist/*          # must say PASSED for both
```

## 2. (Recommended) dry-run on TestPyPI
```bash
python -m twine upload --repository testpypi dist/*
# in a CLEAN venv, verify it installs and runs:
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple cartograph
carto doctor
```

## 3. Publish to real PyPI
```bash
python -m twine upload dist/*
#   username: __token__
#   password: pypi-<your-token>
```
Then anyone can:
```bash
pip install cartograph__v1
carto init && carto ingest ~/code && carto viz
```

## Releasing a new version
1. Bump `version` in `pyproject.toml`.
2. Tag it: `git tag v0.1.1 && git push --tags`.
3. Rebuild + upload (steps 1 + 3).

## Reference packs (graph downloads)
Reference packs are **not** on PyPI — they're zipped graphs attached to GitHub Releases. Build one with
`python scripts/build_reference_pack.py --field <field> --out ./pack` (see REFERENCE_PACKS.md), zip the
output `graph.sqlite` (+ `index/` if you ran `carto index`), and attach it to a Release.
