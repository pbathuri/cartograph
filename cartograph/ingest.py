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
    h = hashlib.sha1()
    h.update(p.read_bytes()[:1_000_000])
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


def infer_field(name: str, blob: str) -> str:
    low = (name + " " + blob[:5000]).lower()
    best, score = "", 0
    for fld, toks in FIELD_TOKENS.items():
        s = sum(1 for t in toks if t in low)
        if s > score:
            best, score = fld, s
    return best if score >= 2 else "general"


def _is_ignored(path: Path, ignore: list[str]) -> bool:
    parts = set(path.parts)
    return any(ig in parts for ig in ignore)


def ingest_path(root: str | Path, store: Store, cfg: Config, *, progress=None) -> IngestStats:
    root = Path(root).expanduser().resolve()
    st = IngestStats()
    if not root.is_dir():
        return st
    # Treat each immediate subdir that looks like a project as a project; else the root itself.
    subdirs = [d for d in root.iterdir() if d.is_dir() and not _is_ignored(d, cfg.ignore)]
    looks_like_container = len(subdirs) >= 2 and not (root / "pyproject.toml").exists() \
        and not (root / "package.json").exists() and not (root / ".git").exists()
    projects = subdirs if looks_like_container else [root]
    max_bytes = int(cfg.max_file_mb * 1_000_000)
    for proj in projects:
        files = [p for p in proj.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXT
                 and not _is_ignored(p, cfg.ignore)]
        if not files:
            continue
        blob = ""
        for p in files[:5]:
            blob += " " + _read(p)[:1000]
        fld = infer_field(proj.name, blob)
        pid = store.upsert_project(proj.name, str(proj), fld)
        st.projects += 1
        st.fields[fld] = st.fields.get(fld, 0) + 1
        for p in files:
            st.files_seen += 1
            try:
                if p.stat().st_size > max_bytes:
                    st.files_skipped += 1
                    continue
            except OSError:
                st.files_skipped += 1
                continue
            fid = store.upsert_file(pid, str(p), p.suffix.lower(), _hash(p))
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
