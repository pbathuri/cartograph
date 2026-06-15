# Reference packs (quantized, public-OSS graph tiers)

A **reference pack** is a pre-built Cartograph graph of *curated public OSS* for a field — a way to
seed a new field without ingesting everything yourself, offered in quantized sizes so you download only
what you need. Packs contain **only public repositories** — never anyone's personal data.

## Sizes (tiers)
| Tier | Rough size | Contents |
|---|---|---|
| **S** (starter) | ~50–150 MB | top 2–3 reference repos for one field, code+docs chunks, FTS only |
| **M** (field) | ~0.5–2 GB | the full reference catalog for a field + semantic index |
| **L** (multi-field) | a few GB | several fields' catalogs + semantic index |

Packs are distributed via **GitHub Releases** (links in the release notes), not committed to the repo,
so the repo stays tiny. Each pack is a zipped `graph.sqlite` (+ optional `index/`).

## Use a pack
```bash
# download a pack zip from Releases, then:
mkdir -p ~/.cartograph && unzip pack_ml_S.zip -d ~/.cartograph
carto stats        # confirm it loaded
carto retrieve "transformer training loop" --chunks
# then add YOUR stuff on top — it merges into the same graph:
carto ingest ~/code
```

## Build your own pack
```bash
# clones the curated public repos for one or more fields and ingests them into a fresh graph
python scripts/build_reference_pack.py --field ml_experiment --out ./pack_ml --depth 1
# optional semantic index:
CARTOGRAPH_HOME=./pack_ml carto index
# then zip ./pack_ml/graph.sqlite (+ index/) and attach to a Release
```

The catalog of public repos per field lives in `cartograph/elite/catalog.py` — edit it to curate your
own pack. Because packs are reproducible from public sources, anyone can rebuild or verify them.
