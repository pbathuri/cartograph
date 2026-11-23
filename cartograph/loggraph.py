"""Log graph — consecutive reasoning, linked by similarity.

A screen log is useful to a later model only if two things are true. The log
itself has to carry more than a caption: the claims it established, what
changed since the previous log, and the questions still open. And the logs
have to be connected, so reasoning can walk.

The walk has a spine, a reasoning parent, and a set of bridges.

  CONTINUES   time order. The previous log on the clock, scored by similarity.
  BUILDS_ON   the parent this log actually continues. Bubble sort lifts the
              strongest similar earlier log to the front, and that one is the
              parent when it clears the bar. Questions and claims compound
              along this edge.
  SIMILAR     the other high-scoring logs from that same sort.

Time order is never rearranged. Bubble sort only ranks which earlier log the
new one is built on. A log that does not resemble its clock neighbor starts a
fresh state, so an unrelated screen does not leak into the next thread.

Usable context compounds along BUILDS_ON. Each node stores a bounded rolling
state, and a walk of those parents is what grows. A claim is kept only when
the text said it, and a question is carried only into a log similar enough
to be the same thread, until that log answers it.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Any, Iterable

from .storage import Store

_STOP = {
    "that", "this", "with", "from", "they", "them", "were", "been", "have",
    "user", "users", "also", "into", "about", "their", "there", "which",
    "would", "could", "should", "while", "where", "when", "what", "your",
    "screen", "working", "activity", "summary", "using", "used",
}
_TOKEN = re.compile(r"[a-z0-9]{4,}")
_SENT = re.compile(r"[^.!?\n]+[.!?]?")
_FILE = re.compile(r"\b[\w./-]+\.(?:py|ipynb|md|tex|pdf|txt|xlsx|csv|json|yaml|yml)\b", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS log_nodes (
    id INTEGER PRIMARY KEY,
    summary_id INTEGER UNIQUE,
    ts REAL,
    prose TEXT,
    claims TEXT,
    decisions TEXT,
    open_questions TEXT,
    delta TEXT,
    carried TEXT,
    answered TEXT,
    entities TEXT,
    state_claims TEXT,
    tokens TEXT
);
CREATE TABLE IF NOT EXISTS log_edges (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    relation TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS idx_log_edges_src ON log_edges(src, relation, score);
"""

# How many earlier logs a new one is compared against, and how many of the
# bubble-sorted winners are kept as knowledge-graph bridges.
_WINDOW = 40
_KEEP = 4
_MIN_SIM = 0.22


def ensure_schema(store: Store) -> None:
    with store.cursor() as c:
        c.executescript(SCHEMA)


def bubble_rank(scored: list[tuple[float, Any]]) -> list[tuple[float, Any]]:
    """Classic bubble sort, highest score first.

    Adjacent items swap when the left one scores lower, so the strongest
    similarity rises to the front of the edge list. This is the whole ranker.
    """
    arr = list(scored)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - 1 - i):
            if arr[j][0] < arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP}


def _sentences(text: str) -> list[str]:
    out = []
    for raw in _SENT.findall(text or ""):
        s = re.sub(r"\s+", " ", raw).strip()
        if len(s.split()) >= 4:
            out.append(s)
    return out


def _near(a: str, b: str) -> bool:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / math.sqrt(len(ta) * len(tb)) >= 0.6


def _answered(question: str, text: str) -> bool:
    q = tokens(question)
    if not q:
        return False
    body = tokens(text)
    return len(q & body) / len(q) >= 0.6


def cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def infer_record(text: str, prior: dict | None = None, *, extra: dict | None = None) -> dict:
    """Pull a bounded, checkable record out of one log, given the prior state.

    Nothing is added that the text did not say. Prior questions survive only
    when this log did not answer them. The rolling state keeps the latest
    claims so the next log continues them instead of starting from zero.
    """
    prose = re.sub(r"\s+", " ", (text or "").strip())
    extra = extra or {}
    stated_claims = [str(c).strip() for c in (extra.get("claims") or []) if str(c).strip()]
    stated_questions = [str(q).strip() for q in (extra.get("open_questions") or []) if str(q).strip()]
    stated_decisions = [str(d).strip() for d in (extra.get("decisions") or []) if str(d).strip()]
    sentences = _sentences(prose)
    claims = stated_claims or [s for s in sentences if "?" not in s][:8]
    questions = stated_questions or [s for s in sentences if "?" in s][:4]
    decisions = stated_decisions or [s for s in claims if re.search(
        r"\b(decided|chose|choosing|will|switching|adopted)\b", s, re.I)][:3]
    entities = []
    for f in _FILE.findall(prose):
        if f not in entities:
            entities.append(f)
    for ent in extra.get("entities") or []:
        if ent and ent not in entities:
            entities.append(str(ent))

    prior = prior or {}
    prior_claims = list(prior.get("state_claims") or [])
    prior_questions = list(prior.get("open_questions") or [])
    delta = [c for c in claims if not any(_near(c, p) for p in prior_claims)]
    if extra.get("delta"):
        delta_line = str(extra["delta"]).strip()
        if delta_line and delta_line not in delta:
            delta = [delta_line, *delta]
    answered = [q for q in prior_questions if _answered(q, prose)]
    carried = [q for q in prior_questions if q not in answered]
    open_questions = []
    for q in carried + questions:
        if q not in open_questions:
            open_questions.append(q)
    open_questions = open_questions[:6]
    state = (prior_claims + delta)[-12:]
    bag = tokens(prose + " " + " ".join(claims))
    return {
        "prose": prose[:1200],
        "claims": claims[:8],
        "decisions": decisions[:3],
        "open_questions": open_questions,
        "delta": delta[:6],
        "carried": carried[:6],
        "answered": answered[:6],
        "entities": entities[:8],
        "state_claims": state,
        "tokens": sorted(bag),
    }


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default
    return val


def _recent(c: sqlite3.Connection, limit: int = _WINDOW) -> list[tuple[int, set[str]]]:
    rows = c.execute(
        "SELECT id, tokens FROM log_nodes ORDER BY ts DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        out.append((int(r["id"]), set(_loads(r["tokens"], []))))
    return out


def link_similar(c: sqlite3.Connection, src: int, src_tokens: set[str],
                 recent: list[tuple[int, set[str]]]) -> int:
    """Score recent logs, bubble the strongest to the front, store those edges."""
    scored = [(cosine(src_tokens, bag), nid) for nid, bag in recent if nid != src and bag]
    ranked = bubble_rank(scored)
    kept = 0
    for score, nid in ranked:
        if score < _MIN_SIM or kept >= _KEEP:
            break
        lo, hi = (src, nid) if src < nid else (nid, src)
        c.execute(
            """INSERT INTO log_edges(src, dst, relation, score) VALUES(?,?, 'SIMILAR', ?)
               ON CONFLICT(src, dst, relation) DO UPDATE SET score=excluded.score""",
            (lo, hi, round(score, 4)),
        )
        kept += 1
    return kept


def ingest_logs(store: Store, rows: Iterable[dict]) -> dict:
    """Turn logs into nodes. Each one builds on its strongest similar predecessor.

    Rows are applied in time order. The similarity window stays in memory.
    Bubble sort picks the parent. The clock predecessor is recorded separately
    and does not inherit state unless it is that parent.
    """
    ensure_schema(store)
    ordered = sorted(rows, key=lambda r: (float(r.get("start") or 0), int(r.get("id") or 0)))
    nodes = edges = 0
    with store.cursor() as c:
        have = {
            int(r[0]) for r in c.execute(
                "SELECT summary_id FROM log_nodes WHERE summary_id IS NOT NULL"
            )
        }
        loaded = c.execute(
            """SELECT id, ts, tokens, state_claims, open_questions
               FROM log_nodes ORDER BY ts DESC, id DESC LIMIT ?""",
            (_WINDOW,),
        ).fetchall()
        states: dict[int, dict] = {}
        recent: list[tuple[int, set[str]]] = []
        for r in loaded:
            bag = set(_loads(r["tokens"], []))
            states[int(r["id"])] = {
                "state_claims": _loads(r["state_claims"], []),
                "open_questions": _loads(r["open_questions"], []),
                "tokens": bag,
            }
            recent.append((int(r["id"]), bag))
        clock_id = int(loaded[0]["id"]) if loaded else None
        clock_ts = float(loaded[0]["ts"] or 0) if loaded else -1.0
        for row in ordered:
            sid = int(row.get("id") or 0) or None
            if sid is not None and sid in have:
                continue
            prose = row.get("summary") or ""
            if not prose.strip():
                continue
            ts = float(row.get("start") or 0)
            draft = tokens(prose)
            if clock_id is not None and ts < clock_ts:
                window = _recent(c)
                clock_for_this = window[0][0] if window else None
            else:
                window = recent
                clock_for_this = clock_id
            ranked = bubble_rank([
                (cosine(draft, bag), nid) for nid, bag in window if bag
            ])
            parent_id = None
            parent_score = 0.0
            if ranked and ranked[0][0] >= _MIN_SIM:
                parent_score, parent_id = ranked[0][0], int(ranked[0][1])
            prior = states.get(parent_id) if parent_id is not None else None
            if parent_id is not None and prior is None:
                parent_row = c.execute(
                    "SELECT state_claims, open_questions FROM log_nodes WHERE id=?",
                    (parent_id,),
                ).fetchone()
                if parent_row is not None:
                    prior = {
                        "state_claims": _loads(parent_row["state_claims"], []),
                        "open_questions": _loads(parent_row["open_questions"], []),
                    }
            extra = {
                "claims": row.get("claims"),
                "decisions": row.get("decisions"),
                "open_questions": row.get("open_questions"),
                "delta": row.get("delta"),
                "entities": row.get("entities"),
            }
            rec = infer_record(prose, prior, extra=extra)
            cur = c.execute(
                """INSERT INTO log_nodes(
                       summary_id, ts, prose, claims, decisions, open_questions, delta,
                       carried, answered, entities, state_claims, tokens)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sid, ts, rec["prose"],
                    json.dumps(rec["claims"]), json.dumps(rec["decisions"]),
                    json.dumps(rec["open_questions"]), json.dumps(rec["delta"]),
                    json.dumps(rec["carried"]), json.dumps(rec["answered"]),
                    json.dumps(rec["entities"]), json.dumps(rec["state_claims"]),
                    json.dumps(rec["tokens"]),
                ),
            )
            nid = int(cur.lastrowid)
            nodes += 1
            if sid is not None:
                have.add(sid)
            bag = set(rec["tokens"])
            if clock_for_this is not None:
                clock_bag = states.get(clock_for_this, {}).get("tokens") or set()
                if not clock_bag:
                    for i, existing in window:
                        if i == clock_for_this:
                            clock_bag = existing
                            break
                c.execute(
                    """INSERT INTO log_edges(src, dst, relation, score)
                       VALUES(?,?, 'CONTINUES', ?)
                       ON CONFLICT(src, dst, relation) DO UPDATE SET score=excluded.score""",
                    (nid, clock_for_this, round(cosine(bag, clock_bag), 4)),
                )
                edges += 1
            edges += link_similar(c, nid, bag, window)
            if parent_id is not None:
                c.execute(
                    """INSERT INTO log_edges(src, dst, relation, score)
                       VALUES(?,?, 'BUILDS_ON', ?)
                       ON CONFLICT(src, dst, relation) DO UPDATE SET score=excluded.score""",
                    (nid, parent_id, round(parent_score, 4)),
                )
                edges += 1
            states[nid] = {
                "state_claims": rec["state_claims"],
                "open_questions": rec["open_questions"],
                "tokens": bag,
            }
            if ts >= clock_ts:
                clock_id = nid
                clock_ts = ts
                recent.insert(0, (nid, bag))
                del recent[_WINDOW:]
    return {"nodes": nodes, "edges": edges}


def _node(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "summary_id": row["summary_id"],
        "prose": row["prose"] or "",
        "claims": _loads(row["claims"], []),
        "delta": _loads(row["delta"], []),
        "open_questions": _loads(row["open_questions"], []),
        "carried": _loads(row["carried"], []),
        "answered": _loads(row["answered"], []),
        "entities": _loads(row["entities"], []),
    }


def assemble(store: Store, prompt: str = "", *, depth: int = 4) -> dict:
    """The chain a model should read: what this continues, then the closest bridges."""
    try:
        with store.cursor() as c:
            exists = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='log_nodes'"
            ).fetchone()
            if not exists:
                return {"has_chain": False}
            q = tokens(prompt)
            anchor = None
            if q:
                best_id = None
                best_score = 0.0
                for r in c.execute("SELECT id, tokens FROM log_nodes"):
                    score = cosine(q, set(_loads(r["tokens"], [])))
                    if score > best_score:
                        best_id, best_score = int(r["id"]), score
                if best_id is not None:
                    anchor = c.execute("SELECT * FROM log_nodes WHERE id=?", (best_id,)).fetchone()
            if anchor is None:
                anchor = c.execute(
                    "SELECT * FROM log_nodes ORDER BY ts DESC, id DESC LIMIT 1"
                ).fetchone()
            if anchor is None:
                return {"has_chain": False}
            chain_rows = [anchor]
            cursor_id = int(anchor["id"])
            seen = {cursor_id}
            for _ in range(depth):
                edge = c.execute(
                    """SELECT dst, score FROM log_edges
                       WHERE src=? AND relation='BUILDS_ON'
                       ORDER BY score DESC LIMIT 1""",
                    (cursor_id,),
                ).fetchone()
                if not edge or edge["score"] < _MIN_SIM or edge["dst"] in seen:
                    break
                nxt = c.execute("SELECT * FROM log_nodes WHERE id=?", (edge["dst"],)).fetchone()
                if nxt is None:
                    break
                chain_rows.append(nxt)
                cursor_id = int(nxt["id"])
                seen.add(cursor_id)
            similar_rows = c.execute(
                """SELECT src, dst, score FROM log_edges
                   WHERE relation='SIMILAR' AND (src=? OR dst=?)
                   ORDER BY score DESC LIMIT ?""",
                (anchor["id"], anchor["id"], _KEEP),
            ).fetchall()
            similar = []
            for e in similar_rows:
                other = e["dst"] if e["src"] == anchor["id"] else e["src"]
                node = c.execute("SELECT * FROM log_nodes WHERE id=?", (other,)).fetchone()
                if node is None:
                    continue
                item = _node(node)
                item["score"] = e["score"]
                similar.append(item)
    except sqlite3.Error:
        return {"has_chain": False}
    chain = [_node(r) for r in chain_rows]
    return {
        "has_chain": True,
        "anchor": chain[0]["prose"][:240],
        "chain": list(reversed(chain)),
        "similar": similar,
        "open_questions": chain[0]["open_questions"],
    }


def render_chain(data: dict) -> list[str]:
    if not data.get("has_chain"):
        return []
    lines = ["Reasoning chain (built from the strongest similar earlier log)"]
    for i, node in enumerate(data.get("chain") or [], 1):
        delta = "; ".join(node.get("delta") or [])[:180]
        lines.append(f"  {i}. {node.get('prose','')[:160]}")
        if delta:
            lines.append(f"     new: {delta}")
    if data.get("open_questions"):
        lines.append("  still open: " + "; ".join(data["open_questions"][:3])[:200])
    if data.get("similar"):
        lines.append("Similar logs (bubble-sorted)")
        for node in data["similar"]:
            lines.append(f"  {node.get('score', 0):.2f}  {node.get('prose','')[:120]}")
    return lines
