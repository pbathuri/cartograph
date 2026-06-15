"""Build a public-OSS reference pack: shallow-clone the curated repos for one or more fields and
ingest them into a fresh Cartograph graph. The result is shareable (public data only) and quantized
by field. See docs/REFERENCE_PACKS.md.

    python scripts/build_reference_pack.py --field ml_experiment --out ./pack_ml --depth 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cartograph.config import Config  # noqa: E402
from cartograph.elite.catalog import CATALOG  # noqa: E402
from cartograph.ingest import ingest_path  # noqa: E402
from cartograph.storage import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", action="append", required=True, help="Field(s) from the catalog. Repeatable.")
    ap.add_argument("--out", type=Path, required=True, help="Output pack directory.")
    ap.add_argument("--depth", type=int, default=1, help="git clone depth (1 = shallow).")
    args = ap.parse_args()

    out = args.out.resolve()
    clones = out / "_repos"
    clones.mkdir(parents=True, exist_ok=True)
    store = Store(out / "graph.sqlite")
    cfg = Config(field_focus=list(args.field))  # the pack IS these fields — label projects accordingly

    repos = []
    for f in args.field:
        if f not in CATALOG:
            print(f"[skip] unknown field: {f} (known: {', '.join(CATALOG)})")
            continue
        repos += [r for r, _, _ in CATALOG[f]]

    for full in repos:
        d = clones / full.replace("/", "__")
        if not (d / ".git").exists():
            print(f"[clone] {full}")
            r = subprocess.run(["git", "clone", "--depth", str(args.depth), "--single-branch",
                                f"https://github.com/{full}", str(d)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [fail] {(r.stderr or '')[:120]}")
                continue
        st = ingest_path(d, store, cfg)
        print(f"[ok] {full}: {st.files_indexed} files, {st.chunks} chunks")

    print(f"\nPack built: {out/'graph.sqlite'}")
    print(f"  stats: {store.stats()}")
    print("  optional semantic index:  CARTOGRAPH_HOME=" + str(out) + " carto index")
    print("  then zip graph.sqlite (+ index/) and attach to a GitHub Release.")


if __name__ == "__main__":
    main()
