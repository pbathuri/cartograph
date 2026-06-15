"""The elite layer — pulls every build toward the frontier of its field, for ANY field.

Curated public-OSS reference catalog, frontier playbooks (the process top practitioners follow),
a domain Definition-of-Done, corpus coverage vs the frontier, and a one-shot `elevate` briefing.
All field-agnostic and extensible: add your own field to catalog.py / playbooks.py / dod.py.
"""
from .catalog import CATALOG, elite_refs, match_field
from .dod import dod_for, score_build
from .elevate import elevate
from .frontier import frontier_report
from .playbooks import PLAYBOOKS, playbook_for

__all__ = ["CATALOG", "elite_refs", "match_field", "dod_for", "score_build",
           "elevate", "frontier_report", "PLAYBOOKS", "playbook_for"]
