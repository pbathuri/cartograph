"""Compass — a constellation of the user's directions, held at once.

Cartograph's graph answers "what do I know?" The persona answers "what do I
respond to?" The compass answers a third question ambitious work actually
needs: "what am I in the middle of — all of it — and which of those threads
does this prompt touch?"

People who explore do not have one active project. They carry slow ambitions,
several ventures, courses, sparks, and short sorties (a case competition, a
grant, a weekend idea) that sometimes grow up and sometimes leave a residue.
A log that keeps a lifetime topic count, or one that keeps only the last two
weeks, both lie. The compass keeps every direction and gives each a horizon:

  ambition   years. Identity. Does not fall off because the week was busy.
  venture    months. A project with artifacts.
  inquiry    a course, a paper, a question being learned.
  sortie     temporary on purpose. Expires from the hot brief. Residue stays.

Status is separate from horizon: spark, exploring, committed, echo, resting,
folded. Echo is the return: a thread that went quiet and came back, so a
model can resume it instead of making the user retell it. Resting threads
stay in memory and stay out of the to-do list, until the prompt names them.

Screen summaries are one source. Declared ambitions (a local ambitions.yaml,
or the PCE user_direction file when you ask the CLI to load it) are another.
Entertainment and pure messaging never become directions.

The steering brief and the MCP `personalize` tool both receive the map, so
every model sees the same constellation.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .storage import Store

DAY = 86400.0
AUTOLOG_DB = Path.home() / "Library/Application Support/ContextD/contextd.sqlite"
USER_DIRECTION = Path.home() / "Tools/project-context-engine/config/user_direction.yaml"

# Whole-token noise: apps, the summarizer's favorite generic labels, filenames
# that mean nothing. Compared after letters-and-digits folding.
_STOP = {
    "arc", "cursor", "claude", "chatgpt", "openai", "outlook", "youtube", "finder",
    "slack", "discord", "whatsapp", "messages", "zoom", "brave", "chrome", "safari",
    "terminal", "iterm", "pdfgear", "github", "windows", "windowmanager", "steam",
    "python", "pytorch", "javascript", "typescript", "readme", "readmemd", "main",
    "index", "app", "test", "tests", "notes", "note", "research", "coding", "writing",
    "other", "watching", "reading", "admin", "review", "design", "communication",
    "user", "project", "projects", "file", "files", "code", "data", "document",
    "email", "mail", "browser", "video", "episode", "screen", "work", "task",
    "general", "unknown", "null", "none", "true", "false",
}

_SORTIE = re.compile(
    r"hackathon|case\s*comp|case\s*competition|scholarship application|"
    r"grant application|one[- ]off|temporary|application draft",
    re.I,
)
_INQUIRY = re.compile(
    r"\bhomework\b|\bassignment\b|\blecture\b|\bsyllabus\b|\bmidterm\b|\bfinal exam\b|"
    r"engr-?e|math-?m|\bcalculus\b|\bfourier\b|\bphasor\b|problem set|\bpset\b",
    re.I,
)
_LIFE = re.compile(
    r"\banime\b|\banimepahe\b|\b9anime\b|\b4anime\b|\bepisode of\b|\bbillions\b|age of empires|\byoutube\b|"
    r"\bnetflix\b|prime video|watching a (video|movie|show)|telugu movie",
    re.I,
)
_STOP_PHRASES = {
    "indianauniversity", "wpsoffice", "zoommeeting", "microsoftteams",
    "microsoftoutlook", "googlechrome", "researchpapers", "windowmanager",
}
_WORK = re.compile(
    r"\bcode\b|\bmodel\b|\bpaper\b|\brepo\b|\bnotebook\b|\bfunction\b|\bapi\b|"
    r"\btrain\b|\bdraft\b|\bfigure\b|\bsimulat",
    re.I,
)

_GUIDANCE = (
    "This person keeps many directions alive at once: slow ambitions, several "
    "ventures, courses, new sparks, and short sorties. Do not collapse them "
    "onto one project, and do not treat a quiet thread as abandoned or a "
    "temporary sortie as an identity. Ambitions outrank the week. Warm items "
    "are all in play together. Sparks are explorations just starting: advance "
    "the one this prompt touches without demanding the others be dropped. "
    "Sorties are temporary on purpose: use what they left behind, do not "
    "restart them as a life plan. Echoes are threads they returned to after "
    "a gap: resume from the residue instead of making them re-explain. "
    "Resting names are memory, not a to-do list, unless one is listed under "
    "relevant because this prompt names it."
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS compass_directions (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE,
    title TEXT,
    horizon TEXT,
    status TEXT,
    declared INTEGER NOT NULL DEFAULT 0,
    first_seen REAL,
    last_seen REAL,
    touch_days INTEGER NOT NULL DEFAULT 0,
    obs_count INTEGER NOT NULL DEFAULT 0,
    echo_count INTEGER NOT NULL DEFAULT 0,
    heat REAL NOT NULL DEFAULT 0,
    residue TEXT,
    artifacts TEXT
);
CREATE TABLE IF NOT EXISTS compass_observations (
    id INTEGER PRIMARY KEY,
    direction_id INTEGER NOT NULL,
    summary_id INTEGER,
    ts REAL NOT NULL,
    lane TEXT,
    note TEXT,
    artifacts TEXT,
    UNIQUE(direction_id, summary_id)
);
CREATE TABLE IF NOT EXISTS compass_kin (
    a INTEGER NOT NULL,
    b INTEGER NOT NULL,
    weight INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS compass_sync (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_summary_id INTEGER NOT NULL DEFAULT 0,
    life_skipped INTEGER NOT NULL DEFAULT 0,
    synced_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_compass_obs_dir ON compass_observations(direction_id, ts);
"""


def ensure_schema(store: Store) -> None:
    with store.cursor() as c:
        c.executescript(SCHEMA)


def _ready(store: Store) -> bool:
    try:
        with store.cursor() as c:
            row = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='compass_directions'"
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _title_case(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return raw
    if raw.isupper() or raw.islower():
        return raw[:1].upper() + raw[1:]
    return raw


def classify_lane(summary: str, activity: str | None, topics: Iterable[str], apps: Iterable[str]) -> str:
    """venture | inquiry | sortie | life. Life never becomes a direction."""
    blob = " ".join([summary or "", activity or "", " ".join(topics or []), " ".join(apps or [])])
    if _SORTIE.search(blob):
        return "sortie"
    if _INQUIRY.search(blob):
        return "inquiry"
    lifeish = _LIFE.search(blob) or (activity or "").lower() in {"watching", "watching videos"}
    if lifeish and not _WORK.search(blob):
        return "life"
    return "venture"


def _mentioned(label: str, summary: str) -> bool:
    """A topic counts only when the summary actually says it.

    The screen summarizer stamps a fixed project list onto unrelated frames.
    Those stamps are not evidence. The longest real word of the label has to
    show up in the prose.
    """
    parts = [canon(p) for p in re.findall(r"[A-Za-z0-9]+", label or "")
             if len(canon(p)) >= 4 and canon(p) not in _STOP]
    if not parts or len(label or "") > 48:
        return False
    blob = canon(summary)
    longest = max(parts, key=len)
    return longest in blob


def extract_keys(topics: Iterable[str], files: Iterable[str], summary: str = "") -> list[tuple[str, str, list[str]]]:
    """Topic names the summary actually uses. Filenames are artifacts, not extra projects.

    A file stem becomes a direction only when the frame named no real topic.
    """
    found: dict[str, str] = {}
    artifacts = [f for f in (files or []) if isinstance(f, str) and f.strip()]

    def take(label: str, *, require_mention: bool) -> None:
        key = canon(label)
        if len(key) < 4 or key in _STOP or key in _STOP_PHRASES:
            return
        if require_mention and not _mentioned(label, summary):
            return
        title = _title_case(label)
        prev = found.get(key)
        if prev is None or len(title) > len(prev):
            found[key] = title

    for label in topics or []:
        take(label, require_mention=True)
    if not found:
        for f in artifacts:
            take(Path(f).stem.replace("_", " ").replace("-", " "), require_mention=False)
    return [(k, title, artifacts) for k, title in found.items()]


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def load_declarations(path: Path | None = None, *, include_user_direction: bool = False) -> list[dict]:
    """Ambitions and declared tracks. Missing files yield an empty list.

    `include_user_direction` reads the PCE direction file when present. The
    library default is off so tests and library calls stay hermetic; the CLI
    turns it on.
    """
    import yaml

    docs: list[tuple[Path, str]] = []
    if path is not None:
        docs.append((path, "ambitions"))
    else:
        from .config import home
        local = home() / "ambitions.yaml"
        if local.exists():
            docs.append((local, "ambitions"))
    if include_user_direction and USER_DIRECTION.exists():
        docs.append((USER_DIRECTION, "direction"))
    out: list[dict] = []
    for p, kind in docs:
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        if kind == "ambitions":
            for item in data.get("ambitions") or []:
                if isinstance(item, dict) and item.get("title"):
                    out.append({
                        "key": canon(item.get("key") or item["title"]),
                        "title": item["title"],
                        "horizon": item.get("horizon") or "ambition",
                        "residue": item.get("residue") or "",
                    })
            for item in data.get("tracks") or []:
                if isinstance(item, dict) and item.get("title"):
                    out.append({
                        "key": canon(item.get("key") or item["title"]),
                        "title": item["title"],
                        "horizon": item.get("horizon") or "venture",
                        "residue": item.get("residue") or "",
                    })
        else:
            profile = data.get("profile") or {}
            goal = profile.get("long_term_goal")
            if goal:
                out.append({
                    "key": "longhorizon",
                    "title": str(goal),
                    "horizon": "ambition",
                    "residue": profile.get("identity_to_protect") or "",
                })
            for track in data.get("active_tracks") or []:
                if not isinstance(track, dict):
                    continue
                title = track.get("label") or track.get("id")
                if not title:
                    continue
                horizon = "ambition" if track.get("starts") else "venture"
                residue = track.get("status") or track.get("why_critical") or ""
                if track.get("starts"):
                    residue = f"starts {track['starts']}. {residue}".strip()
                out.append({
                    "key": canon(str(track.get("id") or title)),
                    "title": str(title),
                    "horizon": horizon,
                    "residue": str(residue),
                })
    # de-dupe by key, first declaration wins
    seen: set[str] = set()
    unique = []
    for item in out:
        if not item["key"] or item["key"] in seen:
            continue
        seen.add(item["key"])
        unique.append(item)
    return unique


def read_autolog_rows(db: Path | None = None, *, since_id: int = 0, limit: int = 50000) -> list[dict]:
    path = db or AUTOLOG_DB
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, startTimestamp, endTimestamp, appNames, summary, keyTopics,
                      activityType, documentPaths
               FROM summaries WHERE id > ? ORDER BY id ASC LIMIT ?""",
            (since_id, limit),
        ).fetchall()
        reasoning: dict[int, sqlite3.Row] = {}
        has_reasoning = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='summary_reasoning'"
        ).fetchone()
        if has_reasoning:
            for rr in conn.execute(
                """SELECT summary_id, claims, decisions, open_questions, delta
                   FROM summary_reasoning WHERE summary_id > ?""",
                (since_id,),
            ):
                reasoning[int(rr["summary_id"])] = rr
    finally:
        conn.close()
    out = []
    for r in rows:
        item = {
            "id": int(r["id"]),
            "start": float(r["startTimestamp"]),
            "end": float(r["endTimestamp"] or r["startTimestamp"]),
            "apps": _loads(r["appNames"], []),
            "summary": r["summary"] or "",
            "topics": _loads(r["keyTopics"], []),
            "activity": r["activityType"],
            "files": _loads(r["documentPaths"], []),
        }
        extra = reasoning.get(item["id"])
        if extra:
            item["claims"] = _loads(extra["claims"], [])
            item["decisions"] = _loads(extra["decisions"], [])
            item["open_questions"] = _loads(extra["open_questions"], [])
            if extra["delta"]:
                item["delta"] = extra["delta"]
        out.append(item)
    return out


def _upsert_direction(c: sqlite3.Connection, key: str, title: str, horizon: str, declared: int,
                      residue: str = "") -> int:
    row = c.execute("SELECT id, title, declared FROM compass_directions WHERE key=?", (key,)).fetchone()
    if row:
        if len(title) > len(row["title"] or "") and not row["declared"]:
            c.execute("UPDATE compass_directions SET title=? WHERE id=?", (title, row["id"]))
        return row["id"]
    cur = c.execute(
        """INSERT INTO compass_directions(key, title, horizon, status, declared, residue, artifacts)
           VALUES(?,?,?,'spark',?,?,?)""",
        (key, title, horizon, declared, residue, "[]"),
    )
    return int(cur.lastrowid)


def _apply_declarations(c: sqlite3.Connection, declarations: list[dict], now: float) -> int:
    n = 0
    for item in declarations:
        key = item["key"]
        row = c.execute("SELECT id FROM compass_directions WHERE key=?", (key,)).fetchone()
        if row:
            c.execute(
                """UPDATE compass_directions
                   SET declared=1, horizon=?, title=?,
                       residue=CASE WHEN ? != '' THEN ? ELSE residue END
                   WHERE id=?""",
                (item["horizon"], item["title"], item.get("residue") or "", item.get("residue") or "", row["id"]),
            )
        else:
            c.execute(
                """INSERT INTO compass_directions(
                       key, title, horizon, status, declared, first_seen, last_seen, residue, artifacts, heat)
                   VALUES(?,?,?,'committed',1,?,?,?, '[]', 0.15)""",
                (key, item["title"], item["horizon"], now, now, item.get("residue") or ""),
            )
            n += 1
    return n


def _derive(horizon: str, declared: int, obs: list, now: float, residue: str = "") -> dict:
    """Status is a function of the clock. Reading the map a month later is not
    stuck with whatever was true on the day it was synced."""
    if not obs:
        return {
            "horizon": horizon,
            "status": "committed" if declared and horizon == "ambition" else "spark",
            "heat": 0.15 if declared else 0.0,
            "echoes": 0,
            "touch_days": 0,
            "first": None,
            "last": None,
            "residue": residue,
            "artifacts": [],
            "obs_count": 0,
        }
    days = {_day(o["ts"]) for o in obs}
    first, last = obs[0]["ts"], obs[-1]["ts"]
    echoes = sum(1 for prev, nxt in zip(obs, obs[1:]) if nxt["ts"] - prev["ts"] >= 21 * DAY)
    arts: list[str] = []
    for o in obs:
        for a in _loads(o["artifacts"], []):
            if a not in arts:
                arts.append(a)
    heat = sum(math.exp(-max(0.0, (now - o["ts"]) / DAY) / 14.0) for o in obs)
    sortie_obs = sum(1 for o in obs if o["lane"] == "sortie")
    inquiry_obs = sum(1 for o in obs if o["lane"] == "inquiry")
    if not declared and sortie_obs > len(obs) / 2:
        horizon = "sortie"
    if horizon == "sortie" and len(days) >= 6 and (last - first) >= 14 * DAY:
        horizon = "venture"  # a temporary effort that kept going became a real thread
    elif horizon == "venture" and inquiry_obs > len(obs) / 2 and sortie_obs == 0:
        horizon = "inquiry"
    age = now - last
    if declared and horizon == "ambition":
        status = "committed"
    elif horizon == "sortie":
        status = "folded" if age > 14 * DAY else ("spark" if len(days) <= 1 else "exploring")
    elif echoes and age < 21 * DAY:
        status = "echo"
    elif age > 21 * DAY and len(days) >= 2:
        status = "resting"
    elif len(days) >= 8 or len(arts) >= 4:
        status = "committed"
    elif len(days) >= 2:
        status = "exploring"
    else:
        status = "spark"
    return {
        "horizon": horizon,
        "status": status,
        "heat": round(heat, 4),
        "echoes": echoes,
        "touch_days": len(days),
        "first": first,
        "last": last,
        "residue": ((obs[-1]["note"] or residue or "")[:280]),
        "artifacts": arts[:12],
        "obs_count": len(obs),
    }


def _recompute(c: sqlite3.Connection, now: float) -> None:
    dirs = c.execute("SELECT * FROM compass_directions").fetchall()
    for d in dirs:
        obs = c.execute(
            "SELECT ts, note, artifacts, lane FROM compass_observations WHERE direction_id=? ORDER BY ts",
            (d["id"],),
        ).fetchall()
        derived = _derive(d["horizon"] or "venture", int(d["declared"] or 0), obs, now, d["residue"] or "")
        if not obs and not (int(d["declared"] or 0) and (d["horizon"] or "") == "ambition"):
            # A declared venture with no screen evidence keeps the status it was inserted with.
            if int(d["declared"] or 0):
                c.execute(
                    "UPDATE compass_directions SET status='committed', heat=0.15, touch_days=0 WHERE id=?",
                    (d["id"],),
                )
            continue
        c.execute(
            """UPDATE compass_directions
               SET horizon=?, status=?, first_seen=COALESCE(?, first_seen), last_seen=COALESCE(?, last_seen),
                   touch_days=?, obs_count=?, echo_count=?, heat=?, residue=?, artifacts=?
               WHERE id=?""",
            (derived["horizon"], derived["status"], derived["first"], derived["last"],
             derived["touch_days"], derived["obs_count"], derived["echoes"], derived["heat"],
             derived["residue"], json.dumps(derived["artifacts"]), d["id"]),
        )


def reset(store: Store) -> None:
    """Drop the constellation so the next sync rebuilds it from scratch."""
    ensure_schema(store)
    from .loggraph import ensure_schema as ensure_logs
    ensure_logs(store)
    with store.cursor() as c:
        c.execute("DELETE FROM compass_kin")
        c.execute("DELETE FROM compass_observations")
        c.execute("DELETE FROM compass_directions")
        c.execute("DELETE FROM compass_sync")
        c.execute("DELETE FROM log_edges")
        c.execute("DELETE FROM log_nodes")


def sync(store: Store, rows: list[dict] | None = None, *, db: Path | None = None,
         declarations: list[dict] | None = None, now: float | None = None,
         load_defaults: bool = False) -> dict:
    """Fold new summaries into the constellation. Idempotent on summary id.

    Pass `rows` in tests. Otherwise summaries are read from the AutoLog
    database, starting after the last synced id.
    """
    ensure_schema(store)
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    decls = list(declarations or [])
    if load_defaults and declarations is None:
        decls = load_declarations(include_user_direction=True)
    with store.cursor() as c:
        state = c.execute("SELECT last_summary_id, life_skipped FROM compass_sync WHERE id=1").fetchone()
        last_id = int(state["last_summary_id"]) if state else 0
        life_skipped = int(state["life_skipped"]) if state else 0
        if rows is None:
            rows = read_autolog_rows(db, since_id=last_id)
        _apply_declarations(c, decls, now)
        added = 0
        skipped_life = 0
        max_id = last_id
        work_rows: list[dict] = []
        for row in rows:
            sid = int(row.get("id") or 0)
            if sid and sid <= last_id:
                continue
            max_id = max(max_id, sid)
            ts = float(row.get("start") or now)
            lane = classify_lane(row.get("summary") or "", row.get("activity"),
                                 row.get("topics") or [], row.get("apps") or [])
            if lane == "life":
                skipped_life += 1
                continue
            work_rows.append(row)
            keys = extract_keys(row.get("topics") or [], row.get("files") or [], row.get("summary") or "")
            if not keys:
                continue
            ids = []
            horizon = {"sortie": "sortie", "inquiry": "inquiry"}.get(lane, "venture")
            note = re.sub(r"\s+", " ", (row.get("summary") or "").strip())[:280]
            arts = [f for f in (row.get("files") or []) if isinstance(f, str)]
            for key, title, _arts in keys:
                did = _upsert_direction(c, key, title, horizon, 0)
                ids.append(did)
                c.execute(
                    """INSERT OR IGNORE INTO compass_observations(
                           direction_id, summary_id, ts, lane, note, artifacts)
                       VALUES(?,?,?,?,?,?)""",
                    (did, sid or None, ts, lane, note, json.dumps(arts[:8])),
                )
                added += 1
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    lo, hi = (a, b) if a < b else (b, a)
                    c.execute(
                        """INSERT INTO compass_kin(a, b, weight) VALUES(?,?,1)
                           ON CONFLICT(a, b) DO UPDATE SET weight = weight + 1""",
                        (lo, hi),
                    )
        _recompute(c, now)
        life_skipped += skipped_life
        c.execute(
            """INSERT INTO compass_sync(id, last_summary_id, life_skipped, synced_at)
               VALUES(1,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 last_summary_id=excluded.last_summary_id,
                 life_skipped=excluded.life_skipped,
                 synced_at=excluded.synced_at""",
            (max_id, life_skipped, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        n_dir = c.execute("SELECT count(*) FROM compass_directions").fetchone()[0]
    from .loggraph import ingest_logs
    graph = ingest_logs(store, work_rows)
    return {
        "directions": n_dir,
        "observations_linked": added,
        "life_skipped": life_skipped,
        "last_summary_id": max_id,
        "log_nodes": graph["nodes"],
        "log_edges": graph["edges"],
    }


def _card(row: sqlite3.Row) -> dict:
    return {
        "key": row["key"],
        "title": row["title"],
        "horizon": row["horizon"],
        "status": row["status"],
        "declared": bool(row["declared"]),
        "first_seen": _iso(row["first_seen"]),
        "last_seen": _iso(row["last_seen"]),
        "touch_days": row["touch_days"],
        "echoes": row["echo_count"],
        "heat": row["heat"],
        "residue": row["residue"] or "",
        "artifacts": _loads(row["artifacts"], []),
    }


def constellation(store: Store, *, now: float | None = None) -> dict:
    """The map models should see. Empty and has_map false until the first sync."""
    if not _ready(store):
        return {"has_map": False, "directions": []}
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    with store.cursor() as c:
        rows = list(c.execute("SELECT * FROM compass_directions"))
        obs_by: dict[int, list] = {}
        for o in c.execute(
            "SELECT direction_id, ts, note, artifacts, lane FROM compass_observations ORDER BY ts"
        ):
            obs_by.setdefault(o["direction_id"], []).append(o)
        life = 0
        state = c.execute("SELECT life_skipped FROM compass_sync WHERE id=1").fetchone()
        if state:
            life = int(state["life_skipped"] or 0)
        kin_rows = c.execute(
            """SELECT d1.title AS a, d2.title AS b, k.weight
               FROM compass_kin k
               JOIN compass_directions d1 ON d1.id = k.a
               JOIN compass_directions d2 ON d2.id = k.b
               ORDER BY k.weight DESC LIMIT 8"""
        ).fetchall()
    cards = []
    for r in rows:
        derived = _derive(r["horizon"] or "venture", int(r["declared"] or 0),
                          obs_by.get(r["id"], []), now, r["residue"] or "")
        card = _card(r)
        card.update({
            "horizon": derived["horizon"],
            "status": derived["status"],
            "first_seen": _iso(derived["first"]) or card["first_seen"],
            "last_seen": _iso(derived["last"]) or card["last_seen"],
            "touch_days": derived["touch_days"],
            "echoes": derived["echoes"],
            "heat": derived["heat"],
            "residue": derived["residue"],
            "artifacts": derived["artifacts"] or card["artifacts"],
        })
        cards.append(card)
    by_status: dict[str, list[dict]] = {}
    for card in cards:
        by_status.setdefault(card["status"], []).append(card)

    def recent(card: dict, days: float) -> bool:
        # last_seen is a date string; compare via the row heat/status instead when missing
        return True if card["last_seen"] is None else True

    ambitions = [c for c in cards if c["horizon"] == "ambition"]
    warm = [c for c in cards
            if c["horizon"] in {"venture", "inquiry"}
            and c["status"] in {"exploring", "committed", "echo"}
            and c["touch_days"] >= 1]
    warm.sort(key=lambda c: c["heat"], reverse=True)
    sparks = [c for c in cards if c["status"] == "spark" and c["horizon"] != "sortie"]
    sparks.sort(key=lambda c: c["heat"], reverse=True)
    echoes = [c for c in cards if c["status"] == "echo"]
    in_flight = [c for c in cards if c["horizon"] == "sortie" and c["status"] != "folded"]
    folded = [c for c in cards if c["horizon"] == "sortie" and c["status"] == "folded"]
    folded.sort(key=lambda c: c["last_seen"] or "", reverse=True)
    resting = [c for c in cards if c["status"] == "resting"]
    resting.sort(key=lambda c: c["last_seen"] or "", reverse=True)
    held = [c for c in cards if c["declared"] and c["horizon"] != "ambition" and c["touch_days"] == 0]
    data = {
        "has_map": bool(cards),
        "as_of": _iso(now),
        "guidance": _GUIDANCE,
        "ambitions": ambitions,
        "warm": warm[:12],
        "sparks": sparks[:8],
        "echoes": echoes[:6],
        "sorties": {"in_flight": in_flight[:6], "folded_residue": folded[:6]},
        "held": held[:8],
        "resting": [
            {"title": c["title"], "last_seen": c["last_seen"], "residue": c["residue"]}
            for c in resting[:12]
        ],
        "kin": [{"a": r["a"], "b": r["b"], "weight": r["weight"]} for r in kin_rows],
        "life_kept_out": life,
        "counts": {
            "directions": len(cards),
            "ambitions": len(ambitions),
            "warm": len(warm),
            "sparks": len(sparks),
            "echoes": len(echoes),
            "resting": len(resting),
            "held": len(held),
            "sorties": len(in_flight) + len(folded),
        },
    }
    try:
        from .loggraph import assemble
        data["reasoning"] = assemble(store, "")
    except Exception:
        data["reasoning"] = {"has_chain": False}
    return data


def _overlap(prompt: str, card: dict) -> int:
    words = {canon(w) for w in re.findall(r"[A-Za-z0-9]{4,}", prompt or "")}
    if not words:
        return 0
    blob = canon(card.get("title", "")) + canon(card.get("residue", ""))
    return sum(1 for w in words if w and w in blob)


def compass_for_prompt(store: Store, prompt: str, *, now: float | None = None) -> dict:
    """Constellation plus the threads this prompt actually touches, including resting ones."""
    data = constellation(store, now=now)
    try:
        from .loggraph import assemble
        data["reasoning"] = assemble(store, prompt)
    except Exception:
        data["reasoning"] = data.get("reasoning") or {"has_chain": False}
    if not data.get("has_map"):
        return data
    pool: list[dict] = []
    for key in ("ambitions", "warm", "sparks", "echoes", "held"):
        pool.extend(data.get(key) or [])
    sorties = data.get("sorties") or {}
    pool.extend(sorties.get("in_flight") or [])
    pool.extend(sorties.get("folded_residue") or [])
    for c in data.get("resting") or []:
        pool.append({**c, "heat": 0, "status": "resting", "horizon": "venture"})
    ranked = []
    seen: set[str] = set()
    for card in pool:
        title = card.get("title") or ""
        if title in seen:
            continue
        score = _overlap(prompt, card)
        if score:
            seen.add(title)
            ranked.append({**card, "match": score})
    ranked.sort(key=lambda c: (c["match"], c.get("heat") or 0), reverse=True)
    data["relevant"] = ranked[:5]
    if ranked:
        lead = ranked[0]["title"]
        data["guidance_line"] = (
            f"Of the live constellation, this prompt touches {lead}. "
            "Answer inside that thread and keep the other warm directions intact."
        )
    else:
        data["guidance_line"] = (
            "No single direction owns this prompt. Answer in a way that can sit "
            "beside the warm threads instead of replacing them."
        )
    return data


def render_text(data: dict) -> str:
    if not data.get("has_map"):
        return "No constellation yet. Run `carto compass --sync` after AutoLog has summaries."
    lines = ["Constellation", ""]
    lines.append(data.get("guidance_line") or data.get("guidance", ""))
    lines.append("")

    def block(title: str, cards: list[dict]) -> None:
        if not cards:
            return
        lines.append(title)
        for c in cards:
            if "horizon" not in c:
                lines.append(f"  {c['title']}  last {c.get('last_seen') or '—'}")
                if c.get("residue"):
                    lines.append(f"    {c['residue'][:160]}")
                continue
            extra = f"  heat {c['heat']:.2f}  days {c['touch_days']}"
            if c.get("echoes"):
                extra += f"  echoes {c['echoes']}"
            lines.append(f"  {c['title']}  [{c['horizon']}/{c['status']}]{extra}")
            if c.get("residue"):
                lines.append(f"    {c['residue'][:180]}")
            if c.get("artifacts"):
                lines.append("    files: " + ", ".join(c["artifacts"][:4]))
        lines.append("")

    block("Ambitions", data.get("ambitions") or [])
    block("Declared, waiting for screen evidence", data.get("held") or [])
    block("Warm", data.get("warm") or [])
    block("Sparks", data.get("sparks") or [])
    block("Echoes — returned after a gap", data.get("echoes") or [])
    sorties = data.get("sorties") or {}
    block("Sorties in flight", sorties.get("in_flight") or [])
    block("Sortie residue", sorties.get("folded_residue") or [])
    block("Resting (memory, not a to-do)", data.get("resting") or [])
    if data.get("relevant"):
        block("Relevant to this prompt", data["relevant"])
    if data.get("kin"):
        lines.append("Often together")
        for k in data["kin"]:
            lines.append(f"  {k['a']}  ×  {k['b']}  ({k['weight']})")
    reasoning = data.get("reasoning") or {}
    if reasoning.get("has_chain"):
        from .loggraph import render_chain
        lines.append("")
        lines.extend(render_chain(reasoning))
    if data.get("life_kept_out"):
        lines.append("")
        lines.append(f"Kept out of the map: {data['life_kept_out']} non-work summaries.")
    return "\n".join(lines).rstrip() + "\n"
