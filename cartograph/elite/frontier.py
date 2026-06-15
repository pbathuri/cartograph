"""Corpus-wide frontier coverage: which fields you actually work in (from ingested project fields),
how much of each field's top-tier reference set you've ingested, and the prioritized acquisition
backlog (weakest frontier first). Read-only."""
from __future__ import annotations

import re

from .catalog import CATALOG

_norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())  # noqa: E731


def frontier_report(store, *, top: int = 8) -> dict:
    keys: set[str] = set()
    active_fields: set[str] = set()
    with store.cursor() as c:
        for r in c.execute("SELECT name, field FROM projects").fetchall():
            d = dict(r)
            nm = d.get("name") or ""
            keys.add(_norm(nm))
            keys.add(_norm(nm.split("/")[-1]))
            if d.get("field"):
                active_fields.add(d["field"])
    fields_out, priority = [], []
    for fld, refs in CATALOG.items():
        active = fld in active_fields
        covered, gaps = [], []
        for repo, teaches, why in refs:
            base = _norm(repo.split("/")[-1])
            if base in keys or _norm(repo) in keys:
                covered.append(repo)
            else:
                gaps.append({"repo": repo, "teaches": teaches, "why": why, "field": fld})
        pct = round(100 * len(covered) / len(refs)) if refs else 0
        rec = {"field": fld, "active": active, "coverage_pct": pct,
               "covered": covered, "gaps": gaps}
        if active or pct > 0:
            fields_out.append(rec)
    for f in sorted([x for x in fields_out if x["active"]], key=lambda x: x["coverage_pct"]):
        priority.extend(f["gaps"])
    fields_out.sort(key=lambda x: -x["coverage_pct"])
    return {"fields": fields_out, "priority_gaps": priority[:top],
            "note": "coverage = top-tier reference repos already ingested; clone+ingest gaps to raise it"}
