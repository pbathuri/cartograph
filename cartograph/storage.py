"""The graph store: one SQLite file holding projects, files, chunks, skills, edges + an FTS5 index.
Local-first, zero servers. The same design that scaled to >1M chunks in the engine Cartograph is
distilled from — but with a clean, dependency-free schema anyone can inspect."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE, path TEXT, field TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT, ext TEXT, hash TEXT,
    UNIQUE(project_id, path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, file_id INTEGER, chunk_text TEXT, ordinal INTEGER
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY, src_id INTEGER, relation TEXT, dst_id INTEGER, metadata TEXT,
    UNIQUE(src_id, relation, dst_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_text, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES ('delete', old.id, old.chunk_text);
END;
CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, relation);
"""


class Store:
    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.cursor() as c:
                c.executescript(SCHEMA)

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        else:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        finally:
            conn.close()

    # ---- writes ----
    def upsert_project(self, name: str, path: str, field: str = "") -> int:
        with self.cursor() as c:
            c.execute("INSERT OR IGNORE INTO projects(name, path, field) VALUES(?,?,?)", (name, path, field))
            if field:
                c.execute("UPDATE projects SET field=? WHERE name=? AND (field='' OR field IS NULL)", (field, name))
            return c.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()[0]

    def upsert_file(self, project_id: int, path: str, ext: str, hash_: str) -> int | None:
        with self.cursor() as c:
            row = c.execute("SELECT id, hash FROM files WHERE project_id=? AND path=?", (project_id, path)).fetchone()
            if row and row["hash"] == hash_:
                return None  # unchanged -> skip
            if row:
                c.execute("DELETE FROM chunks WHERE file_id=?", (row["id"],))
                c.execute("UPDATE files SET hash=? WHERE id=?", (hash_, row["id"]))
                return row["id"]
            cur = c.execute("INSERT INTO files(project_id, path, ext, hash) VALUES(?,?,?,?)",
                            (project_id, path, ext, hash_))
            return cur.lastrowid

    def ensure_file(self, project_id: int, path: str, ext: str = "screen") -> int:
        """Get-or-create a file row WITHOUT touching its chunks (unlike upsert_file, which replaces on
        hash change). Used by the vision pipeline to append screen-activity chunks to a daily 'file'."""
        with self.cursor() as c:
            row = c.execute("SELECT id FROM files WHERE project_id=? AND path=?", (project_id, path)).fetchone()
            if row:
                return row["id"]
            cur = c.execute("INSERT INTO files(project_id, path, ext, hash) VALUES(?,?,?,?)",
                            (project_id, path, ext, "live"))
            return cur.lastrowid

    def recent_chunks(self, project_id: int, limit: int = 5) -> list[dict]:
        """Most-recently-inserted chunks for a project (highest chunk id = newest). For live context."""
        with self.cursor() as c:
            return [dict(r) for r in c.execute(
                "SELECT ch.id AS chunk_id, ch.chunk_text, f.path AS file_path FROM chunks ch "
                "JOIN files f ON f.id = ch.file_id WHERE f.project_id=? ORDER BY ch.id DESC LIMIT ?",
                (project_id, limit)).fetchall()]

    def project_id_by_name(self, name: str) -> int | None:
        with self.cursor() as c:
            row = c.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
            return row["id"] if row else None

    def add_chunks(self, file_id: int, texts: list[str]) -> int:
        with self.cursor() as c:
            c.executemany("INSERT INTO chunks(file_id, chunk_text, ordinal) VALUES(?,?,?)",
                          [(file_id, t, i) for i, t in enumerate(texts)])
        return len(texts)

    def add_edge(self, src_id: int, relation: str, dst_id: int, metadata: str = "") -> None:
        with self.cursor() as c:
            c.execute("INSERT OR IGNORE INTO edges(src_id, relation, dst_id, metadata) VALUES(?,?,?,?)",
                      (src_id, relation, dst_id, metadata))

    def upsert_skill(self, name: str) -> int:
        with self.cursor() as c:
            c.execute("INSERT OR IGNORE INTO skills(name) VALUES(?)", (name,))
            return c.execute("SELECT id FROM skills WHERE name=?", (name,)).fetchone()[0]

    # ---- reads ----
    def search_chunks(self, fts_query: str, limit: int = 20, project_id: int | None = None) -> list[dict]:
        base = (
            "SELECT ch.id AS chunk_id, ch.chunk_text, f.path AS file_path, p.name AS project_name, "
            "p.id AS project_id, chunks_fts.rank AS score FROM chunks_fts "
            "JOIN chunks ch ON ch.id = chunks_fts.rowid JOIN files f ON f.id = ch.file_id "
            "LEFT JOIN projects p ON p.id = f.project_id WHERE chunks_fts MATCH ?"
        )
        params: list[Any] = [fts_query]
        if project_id is not None:
            base += " AND p.id = ?"
            params.append(project_id)
        base += " ORDER BY chunks_fts.rank LIMIT ?"
        params.append(limit)
        with self.cursor() as c:
            return [dict(r) for r in c.execute(base, params).fetchall()]

    def stats(self) -> dict[str, int]:
        out = {}
        with self.cursor() as c:
            for t in ("projects", "files", "chunks", "skills", "edges"):
                try:
                    out[t] = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                except sqlite3.Error:
                    out[t] = 0
        return out

    def graph_sample(self, max_nodes: int = 300) -> dict[str, list]:
        """Project graph for the visualizer: project nodes + shared-skill / relation edges."""
        nodes, edges = [], []
        with self.cursor() as c:
            for r in c.execute("SELECT p.id, p.name, p.field, "
                               "(SELECT count(*) FROM files f WHERE f.project_id=p.id) nf "
                               "FROM projects p LIMIT ?", (max_nodes,)).fetchall():
                d = dict(r)
                nodes.append({"id": f"p{d['id']}", "label": d["name"], "field": d["field"] or "",
                              "size": min(40, 8 + (d["nf"] or 0) // 5), "kind": "project"})
            pid = {n["id"] for n in nodes}
            for r in c.execute("SELECT src_id, relation, dst_id FROM edges "
                               "WHERE relation='PROJECT_RELATED_TO_PROJECT' LIMIT 2000").fetchall():
                s, d = f"p{r['src_id']}", f"p{r['dst_id']}"
                if s in pid and d in pid:
                    edges.append({"source": s, "target": d, "relation": r["relation"]})
        return {"nodes": nodes, "edges": edges}
