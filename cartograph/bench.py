"""`carto bench` — reproducible retrieval numbers anyone can re-run. Always uses a throwaway workspace;
your own graph is never touched.

Suites
  scifact    BEIR SciFact (5,183 abstracts, 300 test queries with relevance labels). Standard public
             benchmark; reports nDCG@10, recall@10/100, MRR@10 and per-query latency for keyword (FTS)
             and, with the semantic extra, hybrid retrieval.
  citations  A generated library of PDFs, Word documents, slide decks, notes and code with planted
             facts on known pages / sections / slides / lines. Measures whether the right file is found
             AND whether the citation points at the right place — the thing most RAG tools don't check.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import statistics
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@contextlib.contextmanager
def isolated_home() -> Iterator[Path]:
    """A temporary CARTOGRAPH_HOME for the duration of a suite."""
    from . import retrieve
    old = os.environ.get("CARTOGRAPH_HOME")
    with tempfile.TemporaryDirectory(prefix="carto-bench-") as d:
        os.environ["CARTOGRAPH_HOME"] = d
        retrieve._INDEX = None
        try:
            yield Path(d)
        finally:
            retrieve._INDEX = None
            if old is None:
                os.environ.pop("CARTOGRAPH_HOME", None)
            else:
                os.environ["CARTOGRAPH_HOME"] = old


def cache_dir() -> Path:
    from .config import home
    d = home() / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- metrics
def ndcg_at(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    pos = {d for d, r in rel.items() if r > 0}
    return len(pos & set(ranked[:k])) / len(pos) if pos else 0.0


def mrr_at(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def _lat(ms: list[float]) -> dict:
    if not ms:
        return {}
    s = sorted(ms)
    return {"p50_ms": round(statistics.median(s), 1), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1)}


# ---------------------------------------------------------------------------------------------- scifact
_HF = "https://huggingface.co/datasets/BeIR"
_HF_ROWS = "https://datasets-server.huggingface.co/rows?dataset=BeIR/scifact&config={c}&split={c}&offset={o}&length=100"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cartograph-bench"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _hf_rows(config: str) -> list[dict]:
    """Rows of a BeIR/scifact config: parquet via pyarrow when installed, else the JSON rows API."""
    try:
        import pyarrow.parquet as pq
        data = _get(f"{_HF}/scifact/resolve/main/{config}/{config}-00000-of-00001.parquet", 120)
        return pq.read_table(io.BytesIO(data)).to_pylist()
    except ImportError:
        pass
    rows, off = [], 0
    while True:
        page = json.loads(_get(_HF_ROWS.format(c=config, o=off)))
        batch = [r["row"] for r in page.get("rows", [])]
        rows += batch
        off += len(batch)
        if not batch or off >= page.get("num_rows_total", 0):
            return rows


def _fetch_scifact(progress=None) -> Path:
    """BEIR SciFact into the bench cache: the official BEIR zip, or the Hugging Face mirror when the
    BEIR host is unreachable (it often is)."""
    root = cache_dir() / "scifact"
    if (root / "corpus.jsonl").exists() and (root / "qrels" / "test.tsv").exists():
        return root
    if progress:
        progress("downloading BEIR SciFact")
    try:
        with zipfile.ZipFile(io.BytesIO(_get(SCIFACT_URL, 20))) as z:
            for name in z.namelist():
                if name.endswith((".jsonl", ".tsv")):
                    out = cache_dir() / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
        return root
    except OSError:
        if progress:
            progress("BEIR host unreachable; using the Hugging Face mirror (BeIR/scifact)")
    (root / "qrels").mkdir(parents=True, exist_ok=True)
    (root / "qrels" / "test.tsv").write_bytes(_get(f"{_HF}/scifact-qrels/resolve/main/test.tsv"))
    for config in ("queries", "corpus"):
        rows = _hf_rows(config)
        with (root / f"{config}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"_id": str(r["_id"]), "title": r.get("title", ""), "text": r.get("text", "")}) + "\n")
    return root


def run_scifact(*, semantic: bool = False, limit: int | None = None, progress=None) -> dict:
    from .config import Config
    from .ingest import ingest_path
    from .retrieve import retrieve
    from .storage import Store
    src = _fetch_scifact(progress)
    qrels: dict[str, dict[str, int]] = {}
    for line in (src / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        q, d, r = line.split("\t")
        qrels.setdefault(q, {})[d] = int(r)
    queries = {}
    for line in (src / "queries.jsonl").read_text().splitlines():
        o = json.loads(line)
        if o["_id"] in qrels:
            queries[o["_id"]] = o["text"]
    qids = sorted(queries)[:limit] if limit else sorted(queries)
    out: dict = {"suite": "scifact", "queries": len(qids)}
    with isolated_home() as h:
        docs = h / "corpus" / "scifact"
        docs.mkdir(parents=True)
        (docs / ".git").mkdir()
        n = 0
        for line in (src / "corpus.jsonl").read_text().splitlines():
            o = json.loads(line)
            (docs / f"{o['_id']}.md").write_text(f"# {o.get('title', '')}\n\n{o.get('text', '')}\n", encoding="utf-8")
            n += 1
        out["documents"] = n
        store, cfg = Store(h / "graph.sqlite"), Config()
        t = time.time()
        st = ingest_path(docs, store, cfg)
        out["ingest_s"] = round(time.time() - t, 1)
        out["chunks"] = st.chunks
        modes = ["fts"]
        if semantic:
            from .embed import available, build_index
            if available():
                if progress:
                    progress(f"embedding {st.chunks:,} chunks for hybrid retrieval")
                t = time.time()
                build_index(store, cfg)
                out["index_s"] = round(time.time() - t, 1)
                modes.append("hybrid")
        for mode in modes:
            if mode == "fts":
                from . import retrieve as R
                R._INDEX = None
                saved = R._semantic_chunk_ids
                R._semantic_chunk_ids = lambda *a, **k: []        # keyword-only run
            ndcg, r10, r100, mrr, lat = [], [], [], [], []
            try:
                for q in qids:
                    t = time.time()
                    res = retrieve(queries[q], store, cfg, top_k=100)
                    lat.append((time.time() - t) * 1000)
                    ranked = list(dict.fromkeys(Path(c["file_path"]).stem for c in res.chunks))
                    ndcg.append(ndcg_at(ranked, qrels[q]))
                    r10.append(recall_at(ranked, qrels[q], 10))
                    r100.append(recall_at(ranked, qrels[q], 100))
                    mrr.append(mrr_at(ranked, qrels[q]))
            finally:
                if mode == "fts":
                    R._semantic_chunk_ids = saved
            out[mode] = {"ndcg@10": round(statistics.mean(ndcg), 4), "recall@10": round(statistics.mean(r10), 4),
                         "recall@100": round(statistics.mean(r100), 4), "mrr@10": round(statistics.mean(mrr), 4),
                         **_lat(lat)}
            if progress:
                progress(f"scifact {mode}: nDCG@10 {out[mode]['ndcg@10']:.3f}")
    return out


# ---------------------------------------------------------------------------------------------- citations
_PROPS = ["corneal residence time", "aqueous humor half-life", "transcorneal flux", "scleral permeability",
          "tear film clearance", "conjunctival uptake", "lens partition coefficient", "vitreous diffusivity"]
_UNITS = ["minutes", "hours", "µg/cm²/h", "cm/s", "µL/min", "%", "", "cm²/s"]
_FILLER = ("Samples were equilibrated at 34 °C and analysed by HPLC with UV detection. Controls followed the "
           "same protocol. Values are means of triplicate runs. ")


def pdf_bytes(pages: list[str]) -> bytes:
    """A minimal valid PDF with one text page per string (Helvetica, ASCII)."""
    def esc(s: str) -> str:
        return s.encode("ascii", "replace").decode().replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = [text[i:i + 90] for i in range(0, len(text), 90)] or [""]
        body = "BT /F1 10 Tf 12 TL 50 750 Td " + " ".join(f"({esc(x)}) Tj T*" for x in lines) + " ET"
        objs.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, b in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{b}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def _docx_bytes(sections: list[tuple[str, str]]) -> bytes:
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    def p(t: str, style: str = "") -> str:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        return f"<w:p>{ppr}<w:r><w:t>{t}</w:t></w:r></w:p>"
    body = "".join(p(h, "Heading1") + p(t) for h, t in sections)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f"<w:document {w}><w:body>{body}</w:body></w:document>")
    return buf.getvalue()


def _pptx_bytes(slides: list[str]) -> bytes:
    a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, t in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", f"<p:sld xmlns:p='p' {a}><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>")
    return buf.getvalue()


def make_citation_corpus(root: Path, seed: int = 7, n_facts: int = 240) -> list[dict]:
    """Write the library and return the ground truth: [{query, file, locator}]."""
    from .documents import pdf_backend
    rng = random.Random(seed)
    proj = root / "lab"
    proj.mkdir(parents=True)
    (proj / ".git").mkdir()
    facts = []
    used = set()
    while len(facts) < n_facts:
        code = f"{rng.choice('ABCDEFGHKLMNPRSTVXZ')}{rng.choice('ABCDEFGHKLMNPRSTVXZ')}-{rng.randint(100, 999)}"
        if code in used:
            continue
        used.add(code)
        k = rng.randrange(len(_PROPS))
        val = round(rng.uniform(0.5, 90), 1)
        facts.append({"code": code, "prop": _PROPS[k],
                      "sentence": f"The {_PROPS[k]} of compound {code} was {val} {_UNITS[k]}.".replace(" .", "."),
                      "query": f"{_PROPS[k]} of {code}"})
    rng.shuffle(facts)
    truth: list[dict] = []
    kinds = ["pdf", "docx", "pptx", "md", "py"] if pdf_backend() else ["docx", "pptx", "md", "py"]
    i, doc = 0, 0
    while i < len(facts):
        kind = kinds[doc % len(kinds)]
        doc += 1
        group = facts[i:i + 6]
        i += len(group)
        name = f"study_{doc:03d}"
        if kind == "pdf":
            pages = [f"Page {j + 1}. {_FILLER}{f['sentence']} {_FILLER}" for j, f in enumerate(group)]
            (proj / f"{name}.pdf").write_bytes(pdf_bytes(pages))
            truth += [{"query": f["query"], "file": f"{name}.pdf", "locator": f"p.{j + 1}"} for j, f in enumerate(group)]
        elif kind == "docx":
            secs = [(f"Experiment {j + 1}", f"{_FILLER}{f['sentence']}") for j, f in enumerate(group)]
            (proj / f"{name}.docx").write_bytes(_docx_bytes(secs))
            truth += [{"query": f["query"], "file": f"{name}.docx", "locator": f"§ Experiment {j + 1}"}
                      for j, f in enumerate(group)]
        elif kind == "pptx":
            (proj / f"{name}.pptx").write_bytes(_pptx_bytes([f"Result {j + 1}: {f['sentence']}" for j, f in enumerate(group)]))
            truth += [{"query": f["query"], "file": f"{name}.pptx", "locator": f"slide {j + 1}"} for j, f in enumerate(group)]
        else:
            lines, where = [], []
            for f in group:
                lines += [("# " if kind == "py" else "") + _FILLER.strip()] * rng.randint(15, 40)
                lines.append(("# " if kind == "py" else "") + f["sentence"])
                where.append(len(lines))
                lines += ["x = 1" if kind == "py" else ""] * rng.randint(5, 20)
            (proj / f"{name}.{kind}").write_text("\n".join(lines) + "\n", encoding="utf-8")
            truth += [{"query": f["query"], "file": f"{name}.{kind}", "line": ln} for f, ln in zip(group, where)]
    return truth


def _cite_ok(t: dict, locator: str | None) -> bool:
    if "line" in t:
        if not locator or not locator.startswith("L"):
            return False
        a, _, b = locator[1:].partition("-")
        return int(a) <= t["line"] <= int(b or a)
    return locator == t["locator"]


def run_citations(*, n_facts: int = 240, progress=None) -> dict:
    from .config import Config
    from .ingest import ingest_path
    from .retrieve import retrieve
    from .storage import Store
    with isolated_home() as h:
        truth = make_citation_corpus(h / "corpus", n_facts=n_facts)
        store, cfg = Store(h / "graph.sqlite"), Config()
        t = time.time()
        st = ingest_path(h / "corpus" / "lab", store, cfg)
        ingest_s = round(time.time() - t, 2)
        hit1 = hit5 = cite1 = 0
        lat = []
        by_kind: dict[str, list[int]] = {}
        for tr in truth:
            t = time.time()
            res = retrieve(tr["query"], store, cfg, top_k=5)
            lat.append((time.time() - t) * 1000)
            names = [Path(c["file_path"]).name for c in res.chunks]
            ok1 = bool(names) and names[0] == tr["file"]
            hit1 += ok1
            hit5 += tr["file"] in names
            good = ok1 and _cite_ok(tr, res.chunks[0].get("locator"))
            cite1 += good
            by_kind.setdefault(Path(tr["file"]).suffix[1:], []).append(int(good))
        n = len(truth)
        out = {"suite": "citations", "queries": n, "documents": st.files_indexed, "chunks": st.chunks,
               "ingest_s": ingest_s, "hit@1": round(hit1 / n, 4), "hit@5": round(hit5 / n, 4),
               "citation@1": round(cite1 / n, 4),
               "citation@1_by_type": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())}, **_lat(lat)}
        if progress:
            progress(f"citations: exact citation@1 {out['citation@1']:.3f}")
        return out


# ---------------------------------------------------------------------------------------------- report
def run(suites: list[str], *, semantic: bool = False, limit: int | None = None, progress=None) -> dict:
    from . import __version__
    report = {"version": __version__, "date": time.strftime("%Y-%m-%d"), "results": []}
    for s in suites:
        if s == "scifact":
            report["results"].append(run_scifact(semantic=semantic, limit=limit, progress=progress))
        elif s == "citations":
            report["results"].append(run_citations(progress=progress))
        else:
            raise ValueError(f"unknown suite {s!r} (scifact, citations)")
    path = cache_dir() / f"results-{report['date']}.json"
    try:
        earlier = json.loads(path.read_text())
        if earlier.get("version") == report["version"]:
            fresh = {r["suite"] for r in report["results"]}
            report["results"] = [r for r in earlier["results"] if r["suite"] not in fresh] + report["results"]
    except (OSError, ValueError, KeyError):
        pass
    path.write_text(json.dumps(report, indent=2))
    report["path"] = str(path)
    return report


def markdown(report: dict) -> str:
    rows = ["| Suite | Mode | Quality | Latency p50 / p95 |", "|---|---|---|---|"]
    for r in report["results"]:
        if r["suite"] == "scifact":
            for mode in ("fts", "hybrid"):
                if mode in r:
                    m = r[mode]
                    rows.append(f"| BEIR SciFact ({r['queries']} queries, {r['documents']:,} docs) | {mode} | "
                                f"nDCG@10 **{m['ndcg@10']:.3f}** · R@100 {m['recall@100']:.3f} · MRR@10 {m['mrr@10']:.3f} | "
                                f"{m['p50_ms']} / {m['p95_ms']} ms |")
        else:
            kinds = ", ".join(f"{k} {v:.2f}" for k, v in r["citation@1_by_type"].items())
            rows.append(f"| Citations ({r['queries']} planted facts, {r['documents']} files) | fts | "
                        f"exact citation@1 **{r['citation@1']:.3f}** · hit@5 {r['hit@5']:.3f} ({kinds}) | "
                        f"{r['p50_ms']} / {r['p95_ms']} ms |")
    return "\n".join(rows) + f"\n\n_cartograph {report['version']}, `carto bench`, {report['date']}_"
