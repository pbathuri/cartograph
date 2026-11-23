"""Agent memory that every client shares: decisions, rejected approaches, preferences, todos and session
logs, kept in the same SQLite file as the graph.

Memories are temporal. Nothing is overwritten: a newer memory *supersedes* an older one, which keeps its
validity interval and stays queryable as history. A memory can be anchored to files; when an anchor's
content changes the memory is reported as possibly stale, not silently trusted. `brief()` assembles a
session-start digest under a hard character budget."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .storage import Store

KINDS = ("decision", "rejected", "preference", "fact", "todo", "note", "session")
_DIRECTIVE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(remember|decision|decided|rejected|todo|preference|note)\s*:\s*(.{4,600})$")
_KIND_OF = {"remember": "note", "decided": "decision"}
_ANCHOR_MAX = 20_000_000


def _now() -> float:
    return time.time()


def _iso(t: float | None) -> str | None:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else None


def _fingerprint(path: str) -> dict | None:
    try:
        p = Path(path).expanduser()
        st = p.stat()
        if not p.is_file() or st.st_size > _ANCHOR_MAX:
            return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}
        return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime,
                "sha1": hashlib.sha1(p.read_bytes()).hexdigest()}
    except OSError:
        return None


def _staleness(anchors: list[dict]) -> list[str]:
    """Why a memory may be out of date: an anchored file changed or disappeared. Files on a volume that
    isn't mounted are reported as unavailable, not missing."""
    out = []
    for a in anchors:
        p = Path(a["path"])
        if not p.exists():
            vol = p.parts[:3] if len(p.parts) > 2 and p.parts[1] == "Volumes" else None
            out.append(f"unavailable (drive not mounted): {p}" if vol and not Path(*vol).exists()
                       else f"anchor missing: {p}")
            continue
        now = _fingerprint(str(p))
        if not now:
            continue
        if a.get("sha1") and now.get("sha1") and a["sha1"] != now["sha1"]:
            out.append(f"anchor changed: {p}")
        elif not a.get("sha1") and (a.get("size") != now.get("size") or a.get("mtime") != now.get("mtime")):
            out.append(f"anchor changed: {p}")
    return out


def project_for(store: Store, project: str | None = None, cwd: str | None = None) -> int | None:
    """A project id from a name, or from the deepest project folder containing `cwd`."""
    if project:
        pid = store.project_id_by_name(project)
        if pid is not None:
            return pid
    if cwd:
        norm = lambda p: os.path.normcase(os.path.normpath(p))  # noqa: E731
        raw = str(Path(cwd).expanduser())
        targets = {norm(raw), norm(os.path.realpath(raw))}   # agents may report a symlinked cwd (/tmp on macOS)
        best, best_len = None, -1
        with store.cursor() as c:
            for pid, path in c.execute("SELECT id, path FROM projects WHERE path IS NOT NULL AND path != ''"):
                root = norm(path)
                inside = any(t == root or t.startswith(root.rstrip(os.sep) + os.sep) for t in targets)
                if inside and len(root) > best_len:
                    best, best_len = pid, len(root)
        return best
    return None


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", s.lower())}


def _row(store: Store, r) -> dict:
    d = dict(r)
    anchors = json.loads(d.get("anchors") or "[]")
    out = {"id": d["id"], "kind": d["kind"], "text": d["text"], "project": d.get("project"),
           "source": d.get("source"), "created": _iso(d["created_at"]), "valid_from": _iso(d["valid_from"]),
           "valid_to": _iso(d.get("valid_to")), "superseded_by": d.get("superseded_by"),
           "confidence": d.get("confidence")}
    if d.get("tags"):
        out["tags"] = d["tags"]
    if anchors:
        out["anchors"] = [a["path"] for a in anchors]
        stale = _staleness(anchors)
        if stale:
            out["stale"] = stale
    if d.get("reason"):
        out["reason"] = d["reason"]
    return out


_SELECT = ("SELECT m.*, p.name AS project FROM memories m LEFT JOIN projects p ON p.id = m.project_id ")


def get(store: Store, mid: int) -> dict | None:
    with store.cursor() as c:
        r = c.execute(_SELECT + "WHERE m.id = ?", (mid,)).fetchone()
    return _row(store, r) if r else None


def remember(store: Store, text: str, kind: str = "note", *, project: str | None = None, cwd: str | None = None,
             anchors: list[str] | None = None, source: str = "cli", session: str | None = None,
             tags: str | None = None, confidence: float = 1.0, supersedes: list[int] | None = None) -> dict:
    """Store a memory. Returns it plus `possible_conflicts`: active memories of the same project that look
    like they talk about the same thing (supersede them if the new one replaces them)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is required")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    pid = project_for(store, project, cwd)
    now = _now()
    with store.cursor() as c:
        dup = c.execute("SELECT id FROM memories WHERE text = ? AND kind = ? AND valid_to IS NULL AND "
                        "project_id IS ?", (text, kind, pid)).fetchone()
    if dup:
        return {**get(store, dup[0]), "duplicate": True, "possible_conflicts": []}
    fps = [fp for fp in (_fingerprint(a) for a in (anchors or [])) if fp]
    with store.cursor() as c:
        cur = c.execute("INSERT INTO memories(kind, text, project_id, source, session, anchors, tags, confidence, "
                        "created_at, valid_from) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (kind, text, pid, source, session, json.dumps(fps) if fps else None, tags, confidence, now, now))
        mid = cur.lastrowid
    for old in supersedes or []:
        supersede(store, int(old), mid, reason="superseded at creation")
    out = get(store, mid) or {"id": mid}
    out["possible_conflicts"] = [] if supersedes else _similar(store, mid, text, kind, pid)
    return out


def _similar(store: Store, mid: int, text: str, kind: str, pid: int | None, threshold: float = 0.3) -> list[dict]:
    if kind in ("session", "todo"):
        return []
    from .retrieve import sanitize
    q = sanitize(text)
    if not q:
        return []
    with store.cursor() as c:
        rows = c.execute(_SELECT + "JOIN memories_fts f ON f.rowid = m.id WHERE memories_fts MATCH ? AND m.id != ? "
                         "AND m.valid_to IS NULL AND m.project_id IS ? AND m.kind IN ('decision','rejected',"
                         "'preference','fact') ORDER BY f.rank LIMIT 8", (q, mid, pid)).fetchall()
    mine = _tokens(text)
    out = []
    for r in rows:
        other = _tokens(r["text"])
        j = len(mine & other) / max(1, len(mine | other))
        if j >= threshold:
            out.append({"id": r["id"], "kind": r["kind"], "text": r["text"], "similarity": round(j, 2)})
    return out


def supersede(store: Store, old_id: int, new_id: int | None = None, reason: str = "") -> dict:
    """End an active memory's validity (optionally pointing at its replacement). History is kept."""
    with store.cursor() as c:
        r = c.execute("SELECT valid_to FROM memories WHERE id = ?", (old_id,)).fetchone()
        if r is None:
            raise ValueError(f"no memory {old_id}")
        if r[0] is not None:
            raise ValueError(f"memory {old_id} is already closed")
        c.execute("UPDATE memories SET valid_to = ?, superseded_by = ?, reason = ? WHERE id = ?",
                  (_now(), new_id, reason or None, old_id))
    return get(store, old_id) or {}


def recall(store: Store, query: str = "", *, project: str | None = None, cwd: str | None = None,
           kinds: list[str] | None = None, history: bool = False, limit: int = 10,
           max_chars: int = 0) -> dict:
    """Memories matching `query` (keyword ranked, newest first on ties), active only unless `history`.
    `max_chars` caps the total text returned."""
    from .retrieve import sanitize
    where, args = [], []
    pid = project_for(store, project, cwd) if (project or cwd) else None
    if pid is not None:
        where.append("m.project_id = ?")
        args.append(pid)
    if not history:
        where.append("m.valid_to IS NULL")
    if kinds:
        where.append(f"m.kind IN ({','.join('?' * len(kinds))})")
        args += list(kinds)
    q = sanitize(query) if query else ""
    sql = _SELECT
    if q:
        sql += "JOIN memories_fts f ON f.rowid = m.id "
        where.insert(0, "memories_fts MATCH ?")
        args.insert(0, q)
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += ("ORDER BY f.rank, m.created_at DESC " if q else "ORDER BY m.created_at DESC ") + "LIMIT ?"
    args.append(int(limit))
    with store.cursor() as c:
        rows = c.execute(sql, args).fetchall()
    items, used = [], 0
    for r in rows:
        item = _row(store, r)
        if max_chars and used + len(item["text"]) > max_chars and items:
            break
        used += len(item["text"])
        items.append(item)
    return {"memories": items, "truncated": len(items) < len(rows), "chars": used}


def brief(store: Store, *, project: str | None = None, cwd: str | None = None, max_chars: int = 2000) -> dict:
    """Session-start digest for the project at `cwd`: decisions, rejected approaches, preferences, open
    todos, recent sessions, and anything gone stale — newest first, never more than `max_chars`."""
    pid = project_for(store, project, cwd)
    scope = "m.project_id = ?" if pid is not None else "m.project_id IS NULL"
    sections = [("Decisions", "decision", 8), ("Rejected approaches (don't retry)", "rejected", 6),
                ("Preferences", "preference", 6), ("Open todos", "todo", 6), ("Facts", "fact", 6),
                ("Recent sessions", "session", 3)]
    lines: list[str] = []
    name = None
    with store.cursor() as c:
        if pid is not None:
            name = c.execute("SELECT name FROM projects WHERE id = ?", (pid,)).fetchone()[0]
        groups = {}
        for title, kind, k in sections:
            rows = c.execute(_SELECT + f"WHERE {scope} AND m.kind = ? AND m.valid_to IS NULL "
                             "ORDER BY m.created_at DESC LIMIT ?", ([pid] if pid is not None else []) + [kind, k]).fetchall()
            groups[title] = [_row(store, r) for r in rows]
        if pid is not None:
            general = c.execute(_SELECT + "WHERE m.project_id IS NULL AND m.kind = 'preference' AND m.valid_to IS NULL "
                                "ORDER BY m.created_at DESC LIMIT 4").fetchall()
            groups["Preferences"] += [_row(store, r) for r in general]
        week = _now() - 7 * 86400
        changed = c.execute(_SELECT + f"WHERE {scope} AND m.valid_to > ? AND m.superseded_by IS NOT NULL "
                            "ORDER BY m.valid_to DESC LIMIT 3", ([pid] if pid is not None else []) + [week]).fetchall()
    header = f"Cartograph memory for {name}" if name else "Cartograph memory"
    lines.append(f"## {header}")
    stale = []
    total = len(lines[0])
    n = 0
    for title, items in groups.items():
        if not items:
            continue
        block = [f"**{title}**"]
        for m in items:
            flag = " (possibly stale)" if m.get("stale") else ""
            block.append(f"- [{m['id']}] {m['text'][:300]}{flag} ({(m['created'] or '')[:10]})")
            if m.get("stale"):
                stale.append(m)
        chunk = "\n".join(block)
        if total + len(chunk) + 1 > max_chars:
            room = max_chars - total - len(block[0]) - 2
            kept = [block[0]]
            for b in block[1:]:
                if room - len(b) - 1 < 0:
                    break
                kept.append(b)
                room -= len(b) + 1
            if len(kept) > 1:
                lines.append("\n".join(kept))
                n += len(kept) - 1
            break
        lines.append(chunk)
        total += len(chunk) + 1
        n += len(items)
    if changed and total < max_chars - 120:
        rep = "**Recently replaced**\n" + "\n".join(f"- [{r['id']}] {r['text'][:120]} → [{r['superseded_by']}]"
                                                    for r in changed)
        if total + len(rep) <= max_chars:
            lines.append(rep)
    text = "\n\n".join(lines) if n else ""
    return {"text": text[:max_chars], "memories": n, "stale": [m["id"] for m in stale], "project": name}


def forget(store: Store, mid: int) -> bool:
    with store.cursor() as c:
        return c.execute("DELETE FROM memories WHERE id = ?", (mid,)).rowcount > 0


def directives(text: str) -> list[tuple[str, str]]:
    """`decision: use pgvector` / `rejected: celery, too heavy` / `todo: ...` lines said in a chat."""
    out = []
    for m in _DIRECTIVE.finditer(text or ""):
        kind = _KIND_OF.get(m.group(1).lower(), m.group(1).lower())
        out.append((kind, m.group(2).strip()))
    return out


def upsert_session(store: Store, source: str, session: str, text: str, *, cwd: str | None = None,
                   anchors: list[str] | None = None) -> int:
    """One `session` memory per (client, session id), rewritten as the session grows."""
    pid = project_for(store, None, cwd)
    fps = [fp for fp in (_fingerprint(a) for a in (anchors or [])[:20]) if fp]
    now = _now()
    with store.cursor() as c:
        r = c.execute("SELECT id FROM memories WHERE kind = 'session' AND source = ? AND session = ?",
                      (source, session)).fetchone()
        if r:
            c.execute("UPDATE memories SET text = ?, anchors = ?, project_id = COALESCE(project_id, ?) WHERE id = ?",
                      (text, json.dumps(fps) if fps else None, pid, r[0]))
            return r[0]
        cur = c.execute("INSERT INTO memories(kind, text, project_id, source, session, anchors, confidence, "
                        "created_at, valid_from) VALUES('session',?,?,?,?,?,?,?,?)",
                        (text, pid, source, session, json.dumps(fps) if fps else None, 1.0, now, now))
        return cur.lastrowid


def stats(store: Store) -> dict[str, Any]:
    if not store.has_table("memories"):
        return {"memories": 0}
    with store.cursor() as c:
        active = c.execute("SELECT count(*) FROM memories WHERE valid_to IS NULL").fetchone()[0]
        total = c.execute("SELECT count(*) FROM memories").fetchone()[0]
    return {"memories": active, "memories_history": total - active}
