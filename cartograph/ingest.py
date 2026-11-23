"""Ingest a folder into the graph: walk files -> chunk text -> store + FTS-index, incrementally.
Re-running only re-processes changed files (hash check). Also infers each project's field and links
projects that share inferred skills, so the graph is more than a pile of documents."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import documents
from .config import Config
from .documents import DOC_EXT, Section
from .storage import Store

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".java", ".go", ".rs", ".rb", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".cu", ".cuh", ".cs", ".php", ".swift", ".kt", ".kts", ".scala", ".r",
    ".jl", ".sql", ".sh", ".bash", ".zsh", ".ps1", ".lua", ".dart", ".m", ".f90", ".zig", ".ex",
    ".exs", ".erl", ".hs", ".ml", ".clj", ".nim", ".pl", ".el", ".proto", ".graphql", ".tf",
    ".md", ".mdx", ".rst", ".txt", ".tex", ".bib", ".org", ".adoc", ".rmd", ".qmd", ".typ",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".css", ".scss", ".vue", ".svelte",
    ".ipynb",
}

# Lightweight field inference from path/content tokens — extend in config.field_focus or here.
FIELD_TOKENS = {
    "ml_experiment": ("torch", "tensorflow", "sklearn", "model", "train", "neural", "transformer", "llm"),
    "quant_research": ("backtest", "alpha", "portfolio", "trading", "factor", "sharpe", "quant"),
    "hpc": ("cuda", "kernel", "mpi", "openmp", "gpu", "simd"),
    "agent_app": ("agent", "prompt", "rag", "langchain", "mcp", "chatbot", "tool_call"),
    "web_frontend": ("react", "vue", "svelte", "component", "css", "tailwind", "nextjs"),
    "data_pipeline": ("airflow", "dbt", "etl", "pipeline", "spark", "kafka", "warehouse"),
    "research_paper": ("abstract", "hypothesis", "experiment", "citation", "arxiv", "theorem"),
    "library": ("__all__", "setup.py", "pyproject", "public api", "semver"),
    "devops": ("dockerfile", "kubernetes", "terraform", "ansible", "ci/cd", "helm"),
    "mobile": ("android", "ios", "swiftui", "kotlin", "flutter", "react native"),
    "game_dev": ("unity", "unreal", "shader", "sprite", "gameobject", "physics"),
    # --- non-software + hardware domains, so non-dev users get real field granularity (not "general") ---
    "embedded_hardware": ("firmware", "rtos", "microcontroller", "fpga", "verilog", "vhdl", "gpio",
                          "i2c", "spi", "uart", "interrupt", "bare-metal", "embedded", "rtl", "soc"),
    "finance_accounting": ("invoice", "ledger", "reconciliation", "gaap", "depreciation", "accrual",
                           "journal entry", "accounts payable", "accounts receivable", "balance sheet",
                           "audit", "deferred tax", "amortization", "payroll", "asc 606", "asc 740"),
    "business_consulting": ("stakeholder", "deliverable", "engagement", "go-to-market", "operating model",
                            "value proposition", "business case", "market sizing", "synergy", "kpi",
                            "roadmap", "due diligence", "swot", "benchmarking"),
    "legal": ("plaintiff", "defendant", "clause", "liability", "statute", "jurisdiction", "indemnification",
              "counterparty", "covenant", "litigation", "compliance memo"),
    "clinical_health": ("patient", "diagnosis", "clinical", "dosage", "symptom", "prognosis", "icd-10",
                        "treatment plan", "cohort", "adverse event"),
    "marketing": ("campaign", "seo", "conversion funnel", "audience segment", "brand", "copywriting",
                  "click-through", "positioning", "messaging"),
}


@dataclass
class IngestStats:
    projects: int = 0
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks: int = 0
    docs: int = 0                 # PDFs / Office / EPUB / HTML indexed this run
    docs_unsupported: int = 0     # PDFs seen but skipped because no PDF backend is installed
    fields: dict[str, int] = field(default_factory=dict)


def _hash(p: Path) -> str:
    """SHA-1 of the whole file, streamed. (Files <= 1 MB hash exactly as the v1 first-MB hash did,
    so upgrading re-chunks only files whose tail could have changed unnoticed.)"""
    h = hashlib.sha1()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_spans(text: str, target: int = 1200, overlap: int = 150) -> list[tuple[int, int]]:
    """(start, end) offsets of each chunk in `text`, whitespace-trimmed. Paragraph-aware with overlap;
    deterministic, no deps. Expects `\\n` line endings."""
    def trimmed(a: int, b: int) -> tuple[int, int] | None:
        seg = text[a:b]
        s = seg.strip()
        if not s:
            return None
        a2 = a + (len(seg) - len(seg.lstrip()))
        return a2, a2 + len(s)

    if len(text) <= target:
        t = trimmed(0, len(text))
        return [t] if t else []
    out, start = [], 0
    while start < len(text):
        end = min(start + target, len(text))
        nl = text.rfind("\n", start + target // 2, end)
        if nl != -1 and end < len(text):
            end = nl
        t = trimmed(start, end)
        if t:
            out.append(t)
        start = max(end - overlap, end) if end >= len(text) else end - overlap
        if start <= 0:
            break
    return out


def chunk_text(text: str, target: int = 1200, overlap: int = 150) -> list[str]:
    text = text.replace("\r\n", "\n")
    return [text[a:b] for a, b in chunk_spans(text, target, overlap)]


def chunk_sections(sections: list[Section], target: int = 1200, overlap: int = 150) -> list[tuple[str, str]]:
    """(chunk text, locator) for every chunk of every section; chunks never straddle sections."""
    out: list[tuple[str, str]] = []
    for sec in sections:
        for a, b in chunk_spans(sec.text, target, overlap):
            out.append((sec.text[a:b], documents.span_locator(sec, a, b)))
    return out


def _sections(p: Path, ocr: bool = False) -> list[Section]:
    ext = p.suffix.lower()
    if ext in DOC_EXT:
        return documents.extract(p, ocr=ocr)
    try:
        if ext == ".ipynb":
            import json
            nb = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            cells = ["".join(c.get("source", [])).replace("\r\n", "\n") for c in nb.get("cells", [])]
            return [documents.notebook_section(cells)]
        return [documents.text_section(p.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n"))]
    except Exception:
        return []


def _read(p: Path) -> str:
    return "\n\n".join(s.text for s in _sections(p))


def wanted(p: Path) -> bool:
    if p.name.startswith(("~$", ".~")):                 # Office/LibreOffice lock files
        return False
    ext = p.suffix.lower()
    return ext in TEXT_EXT or documents.supported(p)


# Strong structural signals — a file extension or manifest is worth several token hits.
EXT_FIELD = {
    ".tsx": "web_frontend", ".jsx": "web_frontend", ".vue": "web_frontend", ".svelte": "web_frontend",
    ".scss": "web_frontend", ".cu": "hpc", ".cuh": "hpc", ".ipynb": "ml_experiment",
    ".tf": "devops", ".swift": "mobile", ".kt": "mobile",
}
MANIFEST_FIELD = {
    "package.json": "web_frontend", "next.config.js": "web_frontend", "tailwind.config.js": "web_frontend",
    "dockerfile": "devops", "docker-compose.yml": "devops", "pubspec.yaml": "mobile",
    "cargo.toml": "library", "go.mod": "library", "dvc.yaml": "research_paper",
    "dbt_project.yml": "data_pipeline", "airflow.cfg": "data_pipeline",
}


def infer_field(name: str, blob: str, exts: dict[str, int] | None = None,
                manifests: set[str] | None = None, focus: list[str] | None = None) -> str:
    """Combine three evidence sources: declared focus (strongest), structural signals (extensions +
    manifest files), and token hits in README/source content. Returns 'general' when genuinely unsure."""
    low = (name + " " + blob[:12000]).lower()
    scores: dict[str, float] = {}

    def add(f: str, w: float) -> None:
        if f:
            scores[f] = scores.get(f, 0.0) + w

    for fld, toks in FIELD_TOKENS.items():
        for t in toks:
            if t in low:
                add(fld, 1.0)
    for ext, n in (exts or {}).items():
        if ext in EXT_FIELD and n >= 2:
            add(EXT_FIELD[ext], 2.5)
    for m in (manifests or set()):
        if m in MANIFEST_FIELD:
            add(MANIFEST_FIELD[m], 3.0)
    # library: a packaged project with a public API + few app entrypoints
    if ("setup.py" in (manifests or set()) or "pyproject.toml" in (manifests or set())) and "__all__" in low:
        add("library", 2.5)
    # declared focus is the user's ground truth — a DOMINANT prior. If a project shows ANY signal for a
    # focus field, prefer it; declared focus should generally win over noisy auto-detection.
    for f in (focus or []):
        add(f, 4.0)
    if not scores:
        return (focus[0] if focus else "general")
    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 2.0:
        return best
    return (focus[0] if focus else "general")


def _is_ignored(path: Path, ignore: list[str]) -> bool:
    parts = set(path.parts)
    return any(ig in parts for ig in ignore)


PROJECT_MARKERS = (".git", "pyproject.toml", "package.json", "setup.py", "Cargo.toml", "go.mod",
                   "pom.xml", "build.gradle", "Project.toml", "DESCRIPTION", "lakefile.lean")
# Folders that hold several projects by convention, even when none of them carries a marker.
CONTAINER_NAMES = {"projects", "repos", "repositories", "references", "workspace", "workspaces"}
MAX_PROJECT_DEPTH = 3


def _subdirs(d: Path, ignore: list[str]) -> list[Path]:
    try:
        return sorted(x for x in d.iterdir() if x.is_dir() and not x.is_symlink()
                      and not x.name.startswith(".") and not _is_ignored(x, ignore))
    except OSError:
        return []


def _has_marker(d: Path) -> bool:
    return any((d / m).exists() for m in PROJECT_MARKERS)


def _marker_within(d: Path, levels: int, ignore: list[str]) -> bool:
    """True if any subfolder up to `levels` deep is a marked project (so `d` is a container of projects)."""
    if levels <= 0:
        return False
    return any(_has_marker(k) or _marker_within(k, levels - 1, ignore) for k in _subdirs(d, ignore))


def discover_projects(root: Path, cfg: Config, *, max_depth: int = MAX_PROJECT_DEPTH) -> list[tuple[Path, bool]]:
    """Find project folders under `root`. Returns (folder, recursive) pairs.

    A folder is a project when it carries a marker (.git, pyproject.toml, ...). A folder is a CONTAINER —
    descended into rather than ingested as one blob — when it holds marked projects within the remaining
    depth, or has a conventional container name ("Projects", "References") and 2+ subfolders.
    A container's own loose files become a shallow (non-recursive) project so nothing is dropped.
    Without this, a folder of 37 cloned repos became a single 2M-chunk project.
    """
    extra = {n.lower() for n in getattr(cfg, "project_containers", []) or []}
    names = CONTAINER_NAMES | extra

    def is_container(d: Path, depth_left: int) -> bool:
        if _has_marker(d):
            return False
        if _marker_within(d, depth_left, cfg.ignore):      # holds marked projects (even just one)
            return True
        return d.name.lower() in names and len(_subdirs(d, cfg.ignore)) >= 2

    def walk(d: Path, depth: int) -> list[tuple[Path, bool]]:
        if depth >= max_depth or not is_container(d, max_depth - depth):
            return [(d, True)]
        out: list[tuple[Path, bool]] = []
        if any(p.is_file() and (p.suffix.lower() in TEXT_EXT or p.suffix.lower() in DOC_EXT)
               for p in d.iterdir()):
            out.append((d, False))
        for k in _subdirs(d, cfg.ignore):
            out.extend(walk(k, depth + 1))
        return out

    if _has_marker(root):
        return [(root, True)]
    subdirs = _subdirs(root, cfg.ignore)
    if len(subdirs) < 2:
        return [(root, True)]
    found: list[tuple[Path, bool]] = []
    for d in subdirs:
        found.extend(walk(d, 1))
    return found


def project_name(root: Path, proj: Path) -> str:
    """Stable, collision-free name: the path relative to the ingest root (first-level folders keep their
    plain name, so existing graphs keep their project names)."""
    try:
        rel = proj.relative_to(root)
    except ValueError:
        return proj.name
    return proj.name if str(rel) in (".", "") else rel.as_posix()


def ingest_path(root: str | Path, store: Store, cfg: Config, *, progress=None, prune: bool = True) -> IngestStats:
    root = Path(root).expanduser().resolve()
    st = IngestStats()
    if not root.is_dir():
        return st
    projects = discover_projects(root, cfg)
    if prune:
        keep = {str(p) for p, _ in projects}
        removed = store.prune_projects_under(str(root), keep)
        if removed and progress:
            progress(f"pruned {removed} stale project(s) under {root}")
    max_bytes = int(cfg.max_file_mb * 1_000_000)
    max_doc_bytes = int(cfg.max_doc_mb * 1_000_000)
    for proj, recursive in projects:
        walker = proj.rglob("*") if recursive else proj.iterdir()
        files = []
        for p in walker:
            if not p.is_file() or _is_ignored(p, cfg.ignore):
                continue
            if wanted(p):
                files.append(p)
            elif p.suffix.lower() == ".pdf":
                st.docs_unsupported += 1
        if not files:
            continue
        text_files = [p for p in files if p.suffix.lower() not in DOC_EXT]
        # Structural signals for field inference: extension histogram + manifest files present.
        exts: dict[str, int] = {}
        manifests: set[str] = set()
        for p in (proj.rglob("*") if recursive else proj.iterdir()):
            if p.is_file() and not _is_ignored(p, cfg.ignore):
                exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
                if p.name.lower() in MANIFEST_FIELD or p.name.lower() in ("setup.py", "pyproject.toml"):
                    manifests.add(p.name.lower())
        # Content sample: READMEs in full + the head of several source files (richer than first-5).
        blob = ""
        readmes = [p for p in text_files if p.name.lower().startswith("readme")]
        for p in readmes[:2]:
            blob += " " + _read(p)[:8000]
        for p in text_files[:20]:
            blob += " " + _read(p)[:800]
        if not text_files:
            blob += " " + " ".join(p.stem for p in files[:200])
        fld = infer_field(proj.name, blob, exts=exts, manifests=manifests, focus=cfg.field_focus)
        pid = store.upsert_project(project_name(root, proj), str(proj), fld)
        st.projects += 1
        st.fields[fld] = st.fields.get(fld, 0) + 1
        for p in files:
            st.files_seen += 1
            is_doc = p.suffix.lower() in DOC_EXT
            try:
                fst = p.stat()
                if fst.st_size > (max_doc_bytes if is_doc else max_bytes):
                    st.files_skipped += 1
                    continue
                prev = store.file_state(pid, str(p))
                if prev and prev.get("size") == fst.st_size and prev.get("mtime") == fst.st_mtime:
                    st.files_skipped += 1                 # same size + mtime: unchanged, don't even read it
                    continue
                digest = _hash(p)
            except OSError:
                st.files_skipped += 1
                continue
            fid = store.upsert_file(pid, str(p), p.suffix.lower(), digest, size=fst.st_size, mtime=fst.st_mtime)
            if fid is None:
                st.files_skipped += 1
                continue
            pairs = chunk_sections(_sections(p, ocr=cfg.ocr))
            if pairs:
                store.add_chunks(fid, [t for t, _ in pairs], locators=[loc for _, loc in pairs])
                st.chunks += len(pairs)
            st.files_indexed += 1
            if is_doc:
                st.docs += 1
            if progress and st.files_indexed % 200 == 0:
                progress(f"{st.files_indexed} files, {st.chunks} chunks ...")
    _link_shared_fields(store)
    return st


_WORD = re.compile(r"[a-z][a-z0-9_]{3,}")


def _link_shared_fields(store: Store) -> None:
    """Cheap relation edges: projects in the same inferred field are related. Gives the graph structure
    without expensive cross-doc similarity; users can train richer links later (carto train)."""
    with store.cursor() as c:
        rows = [dict(r) for r in c.execute("SELECT id, field FROM projects WHERE field != 'general'").fetchall()]
    by_field: dict[str, list[int]] = {}
    for r in rows:
        by_field.setdefault(r["field"], []).append(r["id"])
    for ids in by_field.values():
        for i in range(len(ids)):
            for j in range(i + 1, min(i + 6, len(ids))):  # cap fan-out
                store.add_edge(ids[i], "PROJECT_RELATED_TO_PROJECT", ids[j], "shared_field")
