"""Ingest a folder into the graph: walk files -> chunk text -> store + FTS-index, incrementally.
Re-running only re-processes changed files (hash check). Also infers each project's field and links
projects that share inferred skills, so the graph is more than a pile of documents."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .storage import Store

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".php", ".swift", ".kt", ".scala", ".r", ".jl", ".sql", ".sh", ".ps1",
    ".md", ".rst", ".txt", ".tex", ".org", ".adoc",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".html", ".css", ".scss", ".vue", ".svelte",
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
    fields: dict[str, int] = field(default_factory=dict)


def _hash(p: Path) -> str:
    """SHA-1 of the whole file, streamed. (Files <= 1 MB hash exactly as the v1 first-MB hash did,
    so upgrading re-chunks only files whose tail could have changed unnoticed.)"""
    h = hashlib.sha1()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_text(text: str, target: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking with overlap; deterministic, no deps."""
    text = text.replace("\r\n", "\n")
    if len(text) <= target:
        return [text.strip()] if text.strip() else []
    out, start = [], 0
    while start < len(text):
        end = min(start + target, len(text))
        nl = text.rfind("\n", start + target // 2, end)
        if nl != -1 and end < len(text):
            end = nl
        seg = text[start:end].strip()
        if seg:
            out.append(seg)
        start = max(end - overlap, end) if end >= len(text) else end - overlap
        if start <= 0:
            break
    return out


def _read(p: Path) -> str:
    try:
        if p.suffix.lower() == ".ipynb":
            import json
            nb = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            return "\n\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


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
        if any(p.is_file() and p.suffix.lower() in TEXT_EXT for p in d.iterdir()):
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
    for proj, recursive in projects:
        walker = proj.rglob("*") if recursive else proj.iterdir()
        files = [p for p in walker if p.is_file() and p.suffix.lower() in TEXT_EXT
                 and not _is_ignored(p, cfg.ignore)]
        if not files:
            continue
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
        readmes = [p for p in files if p.name.lower().startswith("readme")]
        for p in readmes[:2]:
            blob += " " + _read(p)[:8000]
        for p in files[:20]:
            blob += " " + _read(p)[:800]
        fld = infer_field(proj.name, blob, exts=exts, manifests=manifests, focus=cfg.field_focus)
        pid = store.upsert_project(project_name(root, proj), str(proj), fld)
        st.projects += 1
        st.fields[fld] = st.fields.get(fld, 0) + 1
        for p in files:
            st.files_seen += 1
            try:
                fst = p.stat()
                if fst.st_size > max_bytes:
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
            chunks = chunk_text(_read(p))
            if chunks:
                store.add_chunks(fid, chunks)
                st.chunks += len(chunks)
            st.files_indexed += 1
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
