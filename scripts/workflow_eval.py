"""Score Cartograph on questions about YOUR files, against ripgrep over the same folders.

    python scripts/workflow_eval.py questions.jsonl            # Cartograph: hybrid, keyword, semantic
    python scripts/workflow_eval.py questions.jsonl --grep     # plus the ripgrep baseline

Each line of questions.jsonl is {"q": "...", "accept": ["path substring", ...], "source": "repo or doc"}.
Write the question the way you would ask it, without copying the file name. A hit is a cited path
that contains one of `accept`; `source` (optional) scores whether the right repo or document shows up
at all. Results are scored on exactly what `retrieve_context` returns (mode chunks, 2,000 tokens).

The grep baseline is generous: files are ranked by how many distinct keywords they contain, then by
match count (an agent normally gets an unranked list). See examples/workflow_questions.jsonl.
"""
from __future__ import annotations

import collections
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _rank(paths: list[str], subs: list[str]) -> int | None:
    return next((i + 1 for i, p in enumerate(paths) if any(s in p for s in subs)), None)


def summarize(label: str, rows: list[dict]) -> dict:
    n = len(rows)
    at = lambda k, key="rank": sum(bool(r[key]) and r[key] <= k for r in rows) / n  # noqa: E731
    s = {"label": label, "n": n, "hit@1": at(1), "hit@5": at(5), "source@5": at(5, "src"),
         "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
         "p50_ms": statistics.median(r["ms"] for r in rows), "median_tokens": statistics.median(r["tokens"] for r in rows)}
    print(f"{label:<9} hit@1 {s['hit@1']:.2f}  hit@5 {s['hit@5']:.2f}  source@5 {s['source@5']:.2f}  "
          f"MRR {s['mrr']:.3f}  p50 {s['p50_ms']:,.0f} ms  tokens {s['median_tokens']:,.0f}", flush=True)
    return s


def cartograph(questions: list[dict]) -> list[dict]:
    from cartograph import retrieve as R
    from cartograph.config import db_path, load_config
    from cartograph.storage import Store
    cfg, store = load_config(), Store(db_path())
    R._semantic_chunk_ids(cfg, "warm up", 1)

    def run(label: str) -> dict:
        rows = []
        for item in questions:
            t = time.perf_counter()
            out = R.context(item["q"], store, cfg, mode="chunks", max_tokens=2000)
            cites = [c.get("cite") or "" for c in out.get("chunks", [])]
            rows.append({"q": item["q"], "rank": _rank(cites, item["accept"]), "tokens": out.get("tokens", 0),
                         "src": _rank(cites, [item.get("source")] if item.get("source") else item["accept"]),
                         "ms": 1000 * (time.perf_counter() - t)})
        return summarize(label, rows)

    real_sem, real_fts = R._semantic_chunk_ids, store.search_chunks
    out = [run("hybrid")]
    R._semantic_chunk_ids = lambda *a, **k: []
    out.append(run("keyword"))
    R._semantic_chunk_ids, store.search_chunks = real_sem, (lambda *a, **k: [])
    out.append(run("semantic"))
    store.search_chunks = real_fts
    return out


def ripgrep(questions: list[dict]) -> dict:
    from cartograph.config import load_config
    from cartograph.retrieve import _STOP, keywords
    roots = [r for r in load_config().roots if Path(r).expanduser().is_dir()]
    rows = []
    for item in questions:
        cmd = ["rg", "-i", "-w", "-o", "--no-line-number", "--with-filename", "--no-heading", "--max-filesize", "2M"]
        for k in (k for k in keywords(item["q"]) if k.lower() not in _STOP):
            cmd += ["-e", k]
        t = time.perf_counter()
        per: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        size = 0
        with subprocess.Popen(cmd + [str(Path(r).expanduser()) for r in roots], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, errors="replace") as p:
            for line in p.stdout:
                size += len(line)
                path, _, match = line.rstrip("\n").rpartition(":")
                per[path][match.lower()] += 1
        ranked = sorted(per, key=lambda f: (-len(per[f]), -sum(per[f].values())))[:50]
        rows.append({"q": item["q"], "rank": _rank(ranked, item["accept"]), "tokens": size // 4, "files": len(per),
                     "src": _rank(ranked, [item.get("source")] if item.get("source") else item["accept"]),
                     "ms": 1000 * (time.perf_counter() - t)})
    s = summarize("ripgrep", rows)
    s["median_files_matched"] = statistics.median(r["files"] for r in rows)
    print(f"          ripgrep matched a median of {s['median_files_matched']:,.0f} files per question")
    return s


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    qs = load(sys.argv[1])
    results = cartograph(qs)
    if "--grep" in sys.argv:
        results.append(ripgrep(qs))
    print(json.dumps(results, indent=1))
