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

# Columns added after v1. Existing graphs gain them in place (ADD COLUMN is O(1) in SQLite); NULL means
# "not recorded yet" and every reader treats it that way.
MIGRATIONS = {
    "files": (("size", "INTEGER"), ("mtime", "REAL")),
    "chunks": (("locator", "TEXT"),),
}


def _migrate(c: sqlite3.Connection) -> None:
    for table, cols in MIGRATIONS.items():
        have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


class Store:
    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.cursor() as c:
                c.executescript(SCHEMA)
                _migrate(c)

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
            c.execute("UPDATE projects SET path=? WHERE name=? AND path IS NOT ?", (path, name, path))
            if field:
                c.execute("UPDATE projects SET field=? WHERE name=? AND (field='' OR field IS NULL)", (field, name))
            return c.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()[0]

    def file_state(self, project_id: int, path: str) -> dict | None:
        """The stored fingerprint of a file (hash, size, mtime), or None if it was never ingested."""
        with self.cursor() as c:
            row = c.execute("SELECT * FROM files WHERE project_id=? AND path=?", (project_id, path)).fetchone()
        return dict(row) if row else None

    def upsert_file(self, project_id: int, path: str, ext: str, hash_: str,
                    size: int | None = None, mtime: float | None = None) -> int | None:
        """Insert or update a file row. Returns the file id when its chunks must be (re)written, or None
        when the content is unchanged (its size/mtime are refreshed so the next run can skip reading it)."""
        with self.cursor() as c:
            row = c.execute("SELECT id, hash FROM files WHERE project_id=? AND path=?", (project_id, path)).fetchone()
            if row and row["hash"] == hash_:
                if size is not None:
                    c.execute("UPDATE files SET size=?, mtime=? WHERE id=?", (size, mtime, row["id"]))
                return None  # unchanged -> skip
            if row:
                c.execute("DELETE FROM chunks WHERE file_id=?", (row["id"],))
                c.execute("UPDATE files SET hash=?, size=?, mtime=? WHERE id=?", (hash_, size, mtime, row["id"]))
                return row["id"]
            cur = c.execute("INSERT INTO files(project_id, path, ext, hash, size, mtime) VALUES(?,?,?,?,?,?)",
                            (project_id, path, ext, hash_, size, mtime))
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

    def delete_project(self, project_id: int) -> None:
        """Remove a project with its files, chunks (FTS rows follow via trigger) and edges."""
        with self.cursor() as c:
            c.execute("DELETE FROM chunks WHERE file_id IN (SELECT id FROM files WHERE project_id=?)", (project_id,))
            c.execute("DELETE FROM files WHERE project_id=?", (project_id,))
            c.execute("DELETE FROM edges WHERE src_id=? OR dst_id=?", (project_id, project_id))
            c.execute("DELETE FROM projects WHERE id=?", (project_id,))

    @staticmethod
    def _under(path: str, root: str) -> bool:
        """Separator- and case-correct 'path is root or inside root' (works on Windows and POSIX)."""
        import os
        a = os.path.normcase(os.path.normpath(path))
        b = os.path.normcase(os.path.normpath(root))
        return a == b or a.startswith(b.rstrip(os.sep) + os.sep)

    def prune_projects_under(self, root: str, keep_paths: set[str]) -> int:
        """Delete projects stored at or under `root` whose folder is no longer a discovered project
        (e.g. an old whole-folder blob after project discovery got finer). Returns how many were removed."""
        import os
        keep = {os.path.normcase(os.path.normpath(k)) for k in keep_paths}
        with self.cursor() as c:
            rows = c.execute("SELECT id, path FROM projects WHERE path IS NOT NULL AND path != ''").fetchall()
        stale = [r["id"] for r in rows if self._under(r["path"], root)
                 and os.path.normcase(os.path.normpath(r["path"])) not in keep]
        for pid in stale:
            self.delete_project(pid)
        return len(stale)

    def prune_outside(self, roots: list[str], keep_names: set[str] | None = None) -> int:
        """Delete projects whose folder is not under any configured root (left over from an old config or
        a folder that was moved away). Projects named in `keep_names` (e.g. live screen activity) stay."""
        keep_names = keep_names or set()
        with self.cursor() as c:
            rows = c.execute("SELECT id, name, path FROM projects").fetchall()
        stale = [r["id"] for r in rows
                 if r["name"] not in keep_names and r["path"]
                 and not any(self._under(r["path"], root) for root in roots)]
        for pid in stale:
            self.delete_project(pid)
        return len(stale)

    def project_id_by_name(self, name: str) -> int | None:
        with self.cursor() as c:
            row = c.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
            return row["id"] if row else None

    def add_chunks(self, file_id: int, texts: list[str], locators: list[str] | None = None) -> int:
        """Append chunks; `locators` (same length) say where each came from: `L12-40`, `p.3`, `slide 2`."""
        locs = locators if locators is not None else [None] * len(texts)
        with self.cursor() as c:
            c.executemany("INSERT INTO chunks(file_id, chunk_text, ordinal, locator) VALUES(?,?,?,?)",
                          [(file_id, t, i, loc or None) for i, (t, loc) in enumerate(zip(texts, locs))])
        return len(texts)

    def has_column(self, table: str, column: str) -> bool:
        """Whether an (older, possibly read-only) graph already has a migrated column."""
        cache = self.__dict__.setdefault("_cols", {})
        if table not in cache:
            with self.cursor() as c:
                cache[table] = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
        return column in cache[table]

    def locator_sql(self, alias: str = "ch") -> str:
        return f"{alias}.locator" if self.has_column("chunks", "locator") else "NULL"

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
            f"SELECT ch.id AS chunk_id, ch.chunk_text, {self.locator_sql()} AS locator, f.path AS file_path, "
            "p.name AS project_name, p.id AS project_id, chunks_fts.rank AS score FROM chunks_fts "
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
