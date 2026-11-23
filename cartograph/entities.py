"""Deterministic entities and relations — no LLM, nothing invented. Every fact points at the chunk it was
read from, says which extractor produced it and how sure that extractor is, and remembers when Cartograph
first and last saw it.

Extracted today: dependencies declared in manifests (requirements, pyproject, package.json, Cargo.toml,
go.mod), imports in Python/JS/TS, papers by DOI and arXiv id, GitHub repositories, Hugging Face models and
datasets. Projects that share specific entities (the same paper, an uncommon library) get linked, weighted
by how rare the shared entity is."""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import PurePath
from typing import Iterable

from .storage import ENTITY_SCHEMA as SCHEMA
from .storage import Store

# (kind, name, relation, confidence, extractor)
Fact = tuple[str, str, str, float, str]


RELATED = "PROJECT_SHARES_ENTITY"

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>(){}\[\],;]+)")
_ARXIV = re.compile(r"(?i)\barxiv(?:\.org/(?:abs|pdf)/|:\s?|\s)(\d{4}\.\d{4,5})(?:v\d+)?")
_GH = re.compile(r"github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9_.-]{1,100})")
_HF = re.compile(r"huggingface\.co/(datasets/)?([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)")
_PY_IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+([A-Za-z_]\w*)[\w.]*[ \t]+import\b|import[ \t]+([A-Za-z_]\w*))", re.M)
_JS_IMPORT = re.compile(r"""(?:\bfrom\s+|\bimport\s+|\brequire\(\s*)['"]((?:@[\w.-]+/)?[\w][\w.-]*)""")
_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~;@]|$)", re.M)
_QUOTED_REQ = re.compile(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~;@][^"']*)?["']""")
_KEY_VERSION = re.compile(r"""^\s*["']?([A-Za-z0-9@][\w@/.-]*)["']?\s*[:=]\s*(?:["']([^"']*)["']|\{)""", re.M)
_VERSIONISH = re.compile(r"^(?:[\^~<>=*]|\d|latest$|workspace:|file:|git\+|github:|npm:|link:)")
_JSON_PAIR = re.compile(r'"([^"\s]+)"\s*:\s*"([^"]*)"')
_GO_REQ = re.compile(r"^\s*(?:require\s+)?([a-z0-9.-]+\.[a-z]{2,}/[\w./-]+)\s+v\d", re.M)

_NOT_DEPS = {"name", "version", "description", "main", "module", "types", "license", "author", "node", "npm",
             "python", "requires-python", "edition", "rust-version", "homepage", "repository", "private",
             "type", "readme", "packagemanager", "target-version", "line-length", "build-backend"}
_PY_ALIASES = {"sklearn": "scikit-learn", "cv2": "opencv-python", "pil": "pillow", "yaml": "pyyaml",
               "bs4": "beautifulsoup4", "skimage": "scikit-image", "dateutil": "python-dateutil",
               "dotenv": "python-dotenv", "attr": "attrs", "fitz": "pymupdf", "google": "google",
               "jwt": "pyjwt", "serial": "pyserial", "crypto": "pycryptodome", "docx": "python-docx",
               "pptx": "python-pptx", "magic": "python-magic"}
_PY_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__", "typing_extensions"}
_NODE_BUILTINS = {"fs", "path", "os", "url", "util", "http", "https", "crypto", "stream", "events", "child_process",
                  "assert", "buffer", "zlib", "net", "tls", "readline", "process", "querystring", "worker_threads",
                  "react-dom/client"}
_TOOLCHAIN = {"pypi/python", "pypi/pip", "pypi/setuptools", "pypi/wheel", "pypi/conda", "pypi/nodejs",
              "pypi/r-base", "pypi/ipykernel", "pypi/ipython", "pypi/jupyter", "pypi/notebook"}
_GH_SKIP = {"orgs", "features", "settings", "marketplace", "topics", "sponsors", "about", "pricing", "login",
            "apps", "collections", "explore", "notifications", "pulls", "issues", "search", "site", "user-attachments"}


def _py_pkg(name: str) -> str:
    n = name.lower().replace("_", "-")
    return "pypi/" + _PY_ALIASES.get(name.lower(), n)


def _manifest_kind(path: str) -> str:
    name = PurePath(path).name.lower()
    if name.startswith("requirements") and name.endswith((".txt", ".in")):
        return "requirements"
    return {"pyproject.toml": "pyproject", "package.json": "npm", "cargo.toml": "cargo", "go.mod": "go",
            "setup.py": "setup", "environment.yml": "conda", "environment.yaml": "conda"}.get(name, "")


_TABLE = re.compile(r"^\s*\[\[?([^\]]+)\]\]?\s*$")
_ARRAY_OPEN = re.compile(r"^\s*([\w.-]+)\s*=\s*\[")
_DEP_ARRAYS = {"dependencies", "requires", "install_requires", "tests_require", "setup_requires", "dev"}
_SPEC = re.compile(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(?:\[[^\]]*\])?\s*[<>=!~][^"']*["']""")


def _py_manifest(text: str, toml: bool) -> Iterable[str]:
    """Requirements inside dependency arrays (PEP 621, setup.py, PDM/uv groups) and Poetry-style tables.
    A chunk may start mid-file, so a quoted requirement with a version spec counts anywhere."""
    section, in_arr = None, False
    for line in text.splitlines():
        t = _TABLE.match(line) if toml else None
        if t:
            section, in_arr = t.group(1).strip().strip('"').lower(), False
            continue
        rest = line
        a = _ARRAY_OPEN.match(line)
        if a:
            key = a.group(1).lower()
            in_arr = key in _DEP_ARRAYS or (section or "").endswith(("optional-dependencies", "dependency-groups",
                                                                     "dev-dependencies"))
            rest = line[a.end():]
        if in_arr:
            for m in _QUOTED_REQ.finditer(rest):
                yield _py_pkg(m.group(1))
            if "]" in rest:
                in_arr = False
        else:
            for m in _SPEC.finditer(rest):
                yield _py_pkg(m.group(1))
            if toml and section and "dependencies" in section:
                k = _KEY_VERSION.match(line)
                if k and k.group(1).lower() not in _NOT_DEPS and (k.group(2) is None or _VERSIONISH.match(k.group(2))):
                    yield _py_pkg(k.group(1))


def _manifest_deps(kind: str, text: str) -> Iterable[str]:
    """Declared dependencies, minus interpreters and tooling that every environment lists."""
    return (d for d in _declared(kind, text) if d not in _TOOLCHAIN)


def _declared(kind: str, text: str) -> Iterable[str]:
    if kind == "requirements":
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith(("-", "git+", "http")):
                m = _REQ_LINE.match(line)
                if m:
                    yield _py_pkg(m.group(1))
    elif kind in ("pyproject", "setup"):
        yield from _py_manifest(text, toml=kind == "pyproject")
    elif kind == "conda":
        in_deps = False
        for line in text.splitlines():
            if re.match(r"^\S", line):
                in_deps = line.startswith("dependencies")
            elif in_deps:
                m = re.match(r"^\s*-\s*([A-Za-z0-9][\w.-]*)\s*(?:[<>=]|$)", line)
                if m and m.group(1) != "pip":
                    yield _py_pkg(m.group(1))
    elif kind == "npm":
        for m in _JSON_PAIR.finditer(text):
            key, val = m.group(1), m.group(2)
            scoped_ok = key.startswith("@") and key.count("/") == 1
            if val is not None and key.lower() not in _NOT_DEPS and _VERSIONISH.match(val) \
                    and ("/" not in key or scoped_ok):
                yield "npm/" + key.lower()
    elif kind == "cargo":
        section = None
        for line in text.splitlines():
            t = _TABLE.match(line)
            if t:
                section = t.group(1).strip().lower()
                continue
            if section is not None and "dependencies" not in section:
                continue
            m = _KEY_VERSION.match(line)
            if m and m.group(1).lower() not in _NOT_DEPS and (m.group(2) is None or _VERSIONISH.match(m.group(2))):
                yield "cargo/" + m.group(1).lower()
    elif kind == "go":
        for m in _GO_REQ.finditer(text):
            yield "go/" + m.group(1)


def extract(text: str, path: str = "", local: frozenset[str] = frozenset()) -> list[Fact]:
    """Facts stated in one chunk of `path`. `local` holds the project's own module names, so importing
    your own `utils` doesn't count as a dependency."""
    out: list[Fact] = []
    ext = PurePath(path).suffix.lower()
    mk = _manifest_kind(path)
    if mk:
        for dep in _manifest_deps(mk, text):
            out.append(("package", dep, "DEPENDS_ON", 1.0, "manifest"))
    if ext in (".py", ".ipynb", ".pyi"):
        for m in _PY_IMPORT.finditer(text):
            mod = m.group(1) or m.group(2)
            if mod and mod not in _PY_STDLIB and mod.lower() not in local:
                out.append(("package", _py_pkg(mod), "IMPORTS", 0.9, "imports"))
    elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte"):
        for m in _JS_IMPORT.finditer(text):
            mod = m.group(1)
            if mod.startswith("node:") or mod in _NODE_BUILTINS:
                continue
            base = "/".join(mod.split("/")[:2]) if mod.startswith("@") else mod.split("/")[0]
            if base.lower() not in local:
                out.append(("package", "npm/" + base.lower(), "IMPORTS", 0.9, "imports"))
    if "10." in text:
        for m in _DOI.finditer(text):
            out.append(("paper", "doi:" + m.group(1).rstrip(".").lower(), "CITES", 0.95, "identifiers"))
    if "rxiv" in text.lower():
        for m in _ARXIV.finditer(text):
            out.append(("paper", "arxiv:" + m.group(1), "CITES", 0.95, "identifiers"))
    if "github.com/" in text:
        for m in _GH.finditer(text):
            owner, repo = m.group(1), re.sub(r"\.git$", "", m.group(2)).rstrip(".")
            if owner.lower() not in _GH_SKIP and repo:
                out.append(("repo", f"github/{owner.lower()}/{repo.lower()}", "REFERENCES", 0.85, "identifiers"))
    if "huggingface.co/" in text:
        for m in _HF.finditer(text):
            kind = "dataset" if m.group(1) else "model"
            if not m.group(2).lower().startswith(("docs/", "spaces/", "blog/", "papers/", "learn/")):
                rel = "USES_DATASET" if kind == "dataset" else "USES_MODEL"
                out.append((kind, "hf/" + m.group(2).lower(), rel, 0.85, "identifiers"))
    return out


# ------------------------------------------------------------------------------------------ storage
def ensure_schema(store: Store) -> bool:
    """True when the graph has the entity tables (a read-only pre-2.0 graph may not)."""
    if store.has_table("mentions"):
        return True
    if store.read_only:
        return False
    with store.cursor() as c:
        c.executescript(SCHEMA)
    store.__dict__.pop("_tables", None)
    return True


def _entity_ids(c, keys: set[tuple[str, str]], cache: dict) -> None:
    missing = [k for k in keys if k not in cache]
    if not missing:
        return
    c.executemany("INSERT OR IGNORE INTO entities(kind, name) VALUES(?,?)", missing)
    for i in range(0, len(missing), 400):
        part = missing[i:i + 400]
        q = " OR ".join("(kind=? AND name=?)" for _ in part)
        for r in c.execute(f"SELECT id, kind, name FROM entities WHERE {q}", [x for k in part for x in k]):
            cache[(r[1], r[2])] = r[0]


def _write(c, rows: dict, cache: dict, now: float) -> None:
    """Additive upsert of {(file_id, kind, name, relation): [count, evidence_chunk, confidence, extractor]}."""
    _entity_ids(c, {(k[1], k[2]) for k in rows}, cache)
    c.executemany(
        "INSERT INTO mentions(entity_id, file_id, relation, count, evidence_chunk, confidence, extractor, "
        "first_seen, last_seen) VALUES(?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(entity_id, file_id, relation) DO UPDATE SET count = mentions.count + excluded.count, "
        "last_seen = excluded.last_seen, confidence = max(mentions.confidence, excluded.confidence), "
        "evidence_chunk = CASE WHEN mentions.count = 0 THEN excluded.evidence_chunk ELSE mentions.evidence_chunk END",
        [(cache[(k[1], k[2])], k[0], k[3], v[0], v[1], v[2], v[3], now, now) for k, v in rows.items()])


def _collect(rows: dict, file_id: int, chunk_id: int, facts: list[Fact]) -> None:
    for kind, name, rel, conf, ext in facts:
        key = (file_id, kind, name, rel)
        cur = rows.get(key)
        if cur is None:
            rows[key] = [1, chunk_id, conf, ext]
        else:
            cur[0] += 1


def local_names(paths: Iterable[str]) -> frozenset[str]:
    """Module names a project defines itself: file stems and folder names of its source files."""
    names: set[str] = set()
    for p in paths:
        pp = PurePath(p)
        if pp.suffix.lower() in (".py", ".js", ".ts", ".tsx", ".jsx", ".mjs"):
            names.add(pp.stem.lower())
            names.update(part.lower() for part in pp.parts[-4:-1])
    return frozenset(names)


def record_file(store: Store, file_id: int, path: str, chunks: list[tuple[int, str]],
                local: frozenset[str] = frozenset()) -> int:
    """Re-extract one file's facts after its chunks changed. Facts it still states keep their first_seen."""
    rows: dict = {}
    for cid, text in chunks:
        _collect(rows, file_id, cid, extract(text, path, local))
    now = time.time()
    with store.cursor() as c:
        c.execute("UPDATE mentions SET count = 0 WHERE file_id = ?", (file_id,))
        if rows:
            _write(c, rows, {}, now)
        c.execute("DELETE FROM mentions WHERE file_id = ? AND count = 0", (file_id,))
    return len(rows)


def backfill(store: Store, progress=None, every: int = 200_000) -> dict:
    """Extract facts for every chunk already in the graph (graphs built before 2.0, or after changing
    extractors). Streams the chunks table once; facts no longer found are dropped, the rest keep their
    first_seen."""
    ensure_schema(store)
    with store.cursor() as c:
        paths = {r[0]: (r[1], r[2]) for r in c.execute("SELECT id, path, project_id FROM files")}
    by_proj: dict = defaultdict(list)
    for p, pid in paths.values():
        by_proj[pid].append(p)
    local = {pid: local_names(ps) for pid, ps in by_proj.items()}
    now = time.time()
    cache: dict = {}
    seen = 0
    rows: dict = {}
    with store.cursor() as w:
        w.execute("UPDATE mentions SET count = 0")
    import sqlite3
    reader = sqlite3.connect(f"file:{store.path}?mode=ro", uri=True, timeout=30)
    try:
        cur = reader.execute("SELECT id, file_id, chunk_text FROM chunks")
        while True:
            batch = cur.fetchmany(5000)
            if not batch:
                break
            for cid, fid, text in batch:
                path, pid = paths.get(fid, ("", None))
                if text:
                    _collect(rows, fid, cid, extract(text, path, local.get(pid, frozenset())))
            seen += len(batch)
            if len(rows) > 200_000 or seen % every < 5000:
                with store.cursor() as w:
                    _write(w, rows, cache, now)
                rows = {}
                if progress:
                    progress(seen)
    finally:
        reader.close()
    with store.cursor() as w:
        if rows:
            _write(w, rows, cache, now)
        w.execute("DELETE FROM mentions WHERE count = 0")
        w.execute("DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT entity_id FROM mentions)")
    links = link_projects(store)
    return {"chunks": seen, **counts(store), "links": links}


def counts(store: Store) -> dict:
    if not store.has_table("mentions"):
        return {"entities": 0, "facts": 0}
    with store.cursor() as c:
        try:
            e = c.execute("SELECT count(*) FROM entities").fetchone()[0]
            m = c.execute("SELECT count(*) FROM mentions").fetchone()[0]
        except Exception:
            return {"entities": 0, "facts": 0}
    return {"entities": e, "facts": m}


def link_projects(store: Store, per_project: int = 5, max_df: int = 40, min_score: float = 3.0) -> int:
    """Link projects that share specific entities. Each shared entity adds log(N/df) (rare = strong); very
    common ones (df > max_df, e.g. numpy) are ignored. A pair qualifies with a high enough score or at least
    two shared entities; each project keeps its strongest few links."""
    if not ensure_schema(store):
        return 0
    with store.cursor() as c:
        n = c.execute("SELECT count(*) FROM projects").fetchone()[0] or 1
        rows = c.execute("SELECT DISTINCT m.entity_id, f.project_id FROM mentions m "
                         "JOIN files f ON f.id = m.file_id WHERE f.project_id IS NOT NULL").fetchall()
        names = {r[0]: f"{r[1]}:{r[2]}" for r in c.execute("SELECT id, kind, name FROM entities")}
    noise = {f"package:{t}" for t in _TOOLCHAIN}
    projs: dict = defaultdict(set)
    for eid, pid in rows:
        if names.get(eid) not in noise:
            projs[eid].add(pid)
    score: dict = defaultdict(float)
    shared: dict = defaultdict(int)
    via: dict = defaultdict(list)
    for eid, ps in projs.items():
        df = len(ps)
        if df < 2 or df > max_df:
            continue
        w = math.log(n / df) + 0.1
        ps = sorted(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                score[(ps[i], ps[j])] += w
                shared[(ps[i], ps[j])] += 1
                via[(ps[i], ps[j])].append((w, names.get(eid, str(eid))))
    best: dict = defaultdict(list)
    for (a, b), s in score.items():
        if s >= min_score or shared[(a, b)] >= 2:
            best[a].append((s, b))
            best[b].append((s, a))
    keep = set()
    for a, lst in best.items():
        for s, b in sorted(lst, reverse=True)[:per_project]:
            keep.add((min(a, b), max(a, b)))
    with store.cursor() as c:
        c.execute("DELETE FROM edges WHERE relation = ?", (RELATED,))
        c.executemany("INSERT OR REPLACE INTO edges(src_id, relation, dst_id, metadata) VALUES(?,?,?,?)",
                      [(a, RELATED, b, json.dumps({
                          "score": round(score[(a, b)], 3),
                          "confidence": round(min(1.0, score[(a, b)] / 15), 3),
                          "via": [nm for _, nm in sorted(via[(a, b)], reverse=True)[:8]]}))
                       for a, b in sorted(keep)])
    return len(keep)


# ------------------------------------------------------------------------------------------ queries
def _iso(t) -> str | None:
    return time.strftime("%Y-%m-%d", time.localtime(t)) if t else None


def project_entities(store: Store, project_id: int, kind: str | None = None, limit: int = 30) -> list[dict]:
    """What a project depends on, imports, cites and references — with evidence."""
    if not ensure_schema(store):
        return []
    q = ("SELECT e.kind, e.name, m.relation, sum(m.count) n, count(DISTINCT m.file_id) files, max(m.confidence) conf, "
         "min(m.first_seen) fs, max(m.last_seen) ls, min(m.evidence_chunk) ev, group_concat(DISTINCT m.extractor) ex "
         "FROM mentions m JOIN entities e ON e.id = m.entity_id JOIN files f ON f.id = m.file_id "
         "WHERE f.project_id = ?" + (" AND e.kind = ?" if kind else "") +
         " GROUP BY e.id, m.relation ORDER BY (m.relation = 'DEPENDS_ON') DESC, files DESC, n DESC LIMIT ?")
    args = [project_id] + ([kind] if kind else []) + [limit]
    with store.cursor() as c:
        rows = c.execute(q, args).fetchall()
    return [_fact_row(store, r) for r in rows]


def find(store: Store, query: str, kind: str | None = None, limit: int = 20) -> list[dict]:
    """Which projects mention an entity (`pydantic`, `1706.03762`, `meta-llama/llama-3`) — with evidence."""
    if not ensure_schema(store):
        return []
    like = f"%{query.lower().strip()}%"
    q = ("SELECT e.kind, e.name, m.relation, p.name project, sum(m.count) n, count(DISTINCT m.file_id) files, "
         "max(m.confidence) conf, min(m.first_seen) fs, max(m.last_seen) ls, min(m.evidence_chunk) ev, "
         "group_concat(DISTINCT m.extractor) ex FROM entities e JOIN mentions m ON m.entity_id = e.id "
         "JOIN files f ON f.id = m.file_id LEFT JOIN projects p ON p.id = f.project_id WHERE e.name LIKE ?"
         + (" AND e.kind = ?" if kind else "") +
         " GROUP BY e.id, m.relation, p.id ORDER BY files DESC, n DESC LIMIT ?")
    args = [like] + ([kind] if kind else []) + [limit]
    with store.cursor() as c:
        rows = c.execute(q, args).fetchall()
    return [_fact_row(store, r) for r in rows]


def _fact_row(store: Store, r) -> dict:
    from .documents import cite
    d = dict(r)
    ev = d.pop("ev", None)
    out = {"entity": f"{d['kind']}:{d['name']}", "relation": d["relation"], "mentions": d["n"],
           "files": d["files"], "confidence": d["conf"], "extractor": d["ex"],
           "first_seen": _iso(d["fs"]), "last_seen": _iso(d["ls"])}
    if "project" in d:
        out["project"] = d["project"]
    if ev:
        with store.cursor() as c:
            row = c.execute(f"SELECT f.path, {store.locator_sql()} FROM chunks ch JOIN files f ON f.id = ch.file_id "
                            "WHERE ch.id = ?", (ev,)).fetchone()
        if row:
            out["evidence"] = {"chunk_id": ev, "cite": cite(row[0], row[1])}
    return out


def related_projects(store: Store, project_id: int) -> list[dict]:
    with store.cursor() as c:
        rows = c.execute("SELECT e.src_id, e.dst_id, e.metadata, p1.name, p2.name FROM edges e "
                         "JOIN projects p1 ON p1.id = e.src_id JOIN projects p2 ON p2.id = e.dst_id "
                         "WHERE e.relation = ? AND (e.src_id = ? OR e.dst_id = ?)",
                         (RELATED, project_id, project_id)).fetchall()
    out = []
    for s, d, meta, n1, n2 in rows:
        m = json.loads(meta or "{}")
        out.append({"project": n2 if s == project_id else n1, **m})
    return sorted(out, key=lambda x: -x.get("score", 0))
