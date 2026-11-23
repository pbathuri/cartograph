"""Frontier scout — pull new research and code for the directions you are actually working on.

Topics come from your own activity, not a fixed list: warm/committed compass directions, sparks, and the
key topics AutoLog saw in PROFESSIONAL activity (coding, research, reading, writing, review) over the last
two weeks. Messaging, video and personal threads never become search terms. For each topic the scout:

  1. queries the arXiv API (relevance-sorted) and GitHub repository search (stars, recently pushed),
  2. writes one small Markdown card per result into ~/.cartograph/frontier/<topic>/ (title, authors, date,
     abstract/description, link, which of your topics it matched),
  3. optionally downloads the top arXiv PDFs to an archive folder (`pdf_dir`, e.g. an external drive),
  4. leaves ingestion to the caller, so cards become searchable context and `frontier_hints` material.

Network access happens only here (an explicit `carto scout` / nightly `carto improve`), never while an
agent is waiting on a brief. Only topic strings are sent; config `scout_exclude` drops any term you never
want to leave the machine, `scout_topics` adds your own.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .config import Config, home
from .storage import Store

UA = "cartograph-scout/1.0 (+https://github.com/pbathuri/cartograph)"
PRO_TYPES = {"coding", "research", "reading", "writing", "review", "design"}
_NOISE = {
    "whatsapp", "youtube", "netflix", "gmail", "outlook", "slack", "discord", "messages", "instagram",
    "linkedin", "twitter", "facebook", "spotify", "travel", "shopping", "chat", "email", "meeting", "call",
    "github", "chrome", "safari", "arc", "cursor", "claude", "chatgpt", "terminal", "finder", "zoom",
    "homework", "exam", "assignment", "quiz", "deadline", "submission", "admin", "project handover",
}


def frontier_dir() -> Path:
    return home() / "frontier"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "topic"


_BASE_LEXICON = {
    "algorithm", "model", "modeling", "modelling", "kernel", "optimization", "optimisation", "regression",
    "neural", "learning", "simulation", "dynamics", "estimation", "inference", "statistics", "statistical",
    "probability", "stochastic", "bayesian", "monte carlo", "numerical", "solver", "matrix", "linear algebra",
    "graph", "network", "signal", "filter", "transform", "fourier", "wavelet", "compiler", "parallel",
    "distributed", "quantum", "protein", "molecular", "cell", "drug", "pharmacokinetic", "pharmacokinetics",
    "retina", "ocular", "cornea", "endothelium", "endothelial", "sedimentation", "rheology", "fpga", "verilog",
    "circuit", "embedded", "dataset", "benchmark", "retrieval", "embedding", "transformer", "llm", "agent",
    "reinforcement", "diffusion", "attention", "classification", "segmentation", "detection", "forecasting",
    "time series", "volatility", "portfolio", "derivatives", "option pricing", "stochastic calculus",
    "factor model", "risk", "econometrics", "control", "physics", "chemistry", "biology", "genomics",
    "eye tracking", "eog", "eeg", "blink", "spmv", "sparse", "gpu", "cuda", "hpc", "mpi", "openmp",
    "convolution", "impulse response", "differential equation", "pde", "ode", "finite element",
    "proof", "theorem", "lean", "topology", "combinatorics", "cryptography", "compression", "quantization",
}


def _lexicon() -> set[str]:
    lex = set(_BASE_LEXICON)
    try:
        from .elite.methods import METHODS
        for m in METHODS.values():
            lex.update(k.lower() for k in m["kw"])
    except Exception:
        pass
    try:
        from .ingest import FIELD_TOKENS
        for toks in FIELD_TOKENS.values():
            lex.update(t.lower() for t in toks if len(t) >= 4)
    except Exception:
        pass
    return lex


_LEX: set[str] | None = None


def is_technical(topic: str) -> bool:
    """A topic leaves the machine only if it names a technical concept (whole-word lexicon match).
    Project codenames, games, people, places and personal questions don't match and are dropped."""
    global _LEX
    if _LEX is None:
        _LEX = _lexicon()
    low = topic.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(k)}s?(?![a-z])", low) for k in _LEX)


def _clean_topic(t: str) -> str | None:
    t = re.sub(r"\s+", " ", str(t)).strip()
    low = t.lower()
    if not (4 <= len(t) <= 60) or "@" in t or "/" in t or "\\" in t:
        return None
    if re.search(r"\.[a-z0-9]{1,5}$", low) or re.search(r"\d{4}-\d{2}", low):   # filenames, dated items
        return None
    if not is_technical(t):
        return None
    if sum(ch.isdigit() for ch in t) > len(t) // 3:
        return None
    if any(w in low.split() or w == low for w in _NOISE):
        return None
    return t


def gather_topics(store: Store, cfg: Config, *, days: float = 14, limit: int = 8, autolog_db=None) -> list[str]:
    exclude = {e.lower() for e in getattr(cfg, "scout_exclude", []) or []}
    manual = [t for t in getattr(cfg, "scout_topics", []) or [] if t]
    counts: Counter = Counter()
    try:
        from .compass import constellation
        cons = constellation(store)
        for c in cons.get("warm", [])[:6] + cons.get("sparks", [])[:4]:
            counts[c["title"]] += 3
    except Exception:
        pass
    try:
        from .graphnet import read_activity
        cutoff = time.time() - days * 86400
        for a in read_activity(autolog_db):
            if a["ts"] < cutoff:
                break
            if a["type"] not in PRO_TYPES:
                continue
            for t in a["topics"][:6]:
                counts[t] += 1
    except Exception:
        pass
    out: list[str] = []
    seen: set[str] = set()
    manual_set = set(manual)
    for t in manual + [t for t, n in counts.most_common(200) if n >= 2]:
        ct = str(t).strip() if t in manual_set else _clean_topic(t)      # your own topics are kept as given
        if not ct or ct.lower() in exclude or slug(ct) in seen:
            continue
        seen.add(slug(ct))
        out.append(ct)
        if len(out) >= limit:
            break
    return out


def _get(url: str, *, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https hosts below
        return r.read()


def arxiv_search(topic: str, n: int = 5) -> list[dict]:
    q = urllib.parse.urlencode({"search_query": f'all:"{topic}"', "sortBy": "relevance", "max_results": n})
    raw = _get(f"https://export.arxiv.org/api/query?{q}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(raw).findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append({
            "source": "arxiv", "id": aid,
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(),
            "summary": re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip(),
            "published": (e.findtext("a:published", "", ns) or "")[:10],
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:6],
            "url": f"https://arxiv.org/abs/{aid}", "pdf": f"https://arxiv.org/pdf/{aid}",
        })
    return out


def github_search(topic: str, n: int = 3) -> list[dict]:
    since = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 365 * 86400))
    q = urllib.parse.urlencode({"q": f"{topic} pushed:>{since}", "sort": "stars", "order": "desc", "per_page": n})
    data = json.loads(_get(f"https://api.github.com/search/repositories?{q}"))
    return [{"source": "github", "id": it["full_name"], "title": it["full_name"],
             "summary": it.get("description") or "", "stars": it.get("stargazers_count", 0),
             "published": (it.get("pushed_at") or "")[:10], "url": it["html_url"]}
            for it in data.get("items", [])[:n]]


def _card(item: dict, topic: str) -> str:
    lines = [f"# {item['title']}", "", f"- source: {item['source']}", f"- matched topic: {topic}",
             f"- date: {item.get('published', '')}", f"- link: {item['url']}"]
    if item.get("authors"):
        lines.append(f"- authors: {', '.join(item['authors'])}")
    if item.get("stars") is not None and item["source"] == "github":
        lines.append(f"- stars: {item['stars']}")
    lines += ["", item.get("summary", "")]
    return "\n".join(lines) + "\n"


def scout(store: Store, cfg: Config, *, topics: list[str] | None = None, per_topic: int = 4,
          repos_per_topic: int = 2, pdfs_per_topic: int = 1, pdf_dir: Path | None = None,
          pause: float = 3.0, progress=None) -> dict:
    topics = topics if topics is not None else gather_topics(store, cfg)
    root = frontier_dir()
    root.mkdir(parents=True, exist_ok=True)
    stats = {"topics": topics, "papers": 0, "repos": 0, "pdfs": 0, "errors": []}
    for t in topics:
        d = root / slug(t)
        d.mkdir(exist_ok=True)
        for fn, n, key in ((arxiv_search, per_topic, "papers"), (github_search, repos_per_topic, "repos")):
            try:
                items = fn(t, n)
            except Exception as e:  # network/rate limit: keep going, report
                stats["errors"].append(f"{fn.__name__}({t}): {e}")
                items = []
            for i, it in enumerate(items):
                name = f"{it['source']}_{slug(it['id'])}.md"
                p = d / name
                if not p.exists():
                    p.write_text(_card(it, t), encoding="utf-8")
                    stats[key] += 1
                if it["source"] == "arxiv" and pdf_dir and i < pdfs_per_topic:
                    out = Path(pdf_dir) / slug(t) / f"{slug(it['id'])}.pdf"
                    if not out.exists():
                        try:
                            out.parent.mkdir(parents=True, exist_ok=True)
                            out.write_bytes(_get(it["pdf"], timeout=60))
                            stats["pdfs"] += 1
                        except Exception as e:
                            stats["errors"].append(f"pdf {it['id']}: {e}")
            time.sleep(pause)                                   # arXiv asks for >= 3 s between calls
        if progress:
            progress(f"scouted '{t}'")
    (root / "_last_run.json").write_text(json.dumps({**stats, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    return stats


def frontier_cards(store: Store, prompt: str, *, limit: int = 3) -> list[dict]:
    """Frontier cards relevant to a prompt (FTS over ingested cards only). Offline, fast."""
    from .retrieve import sanitize
    san = sanitize(prompt)
    if not san:
        return []
    root = str(frontier_dir())
    with store.cursor() as c:
        rows = c.execute(
            "SELECT ch.chunk_text, f.path FROM chunks_fts JOIN chunks ch ON ch.id=chunks_fts.rowid "
            "JOIN files f ON f.id=ch.file_id WHERE chunks_fts MATCH ? AND f.path LIKE ? "
            "ORDER BY chunks_fts.rank LIMIT ?", (san, root + "%", limit * 3)).fetchall()
    out, seen = [], set()
    for r in rows:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        txt = r["chunk_text"] or ""
        title = txt.split("\n", 1)[0].lstrip("# ").strip()
        link = re.search(r"- link: (\S+)", txt)
        topic = re.search(r"- matched topic: (.+)", txt)
        body = txt.split("\n\n", 1)[-1].strip().replace("\n", " ")
        out.append({"title": title[:140], "link": link.group(1) if link else "",
                    "topic": topic.group(1).strip() if topic else "", "gist": body[:220]})
        if len(out) >= limit:
            break
    return out
