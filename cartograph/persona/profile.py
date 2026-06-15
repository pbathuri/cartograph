"""PersonaProfile — the user's model: field weights (which domains matter, and how much), explicit
preferences (style/verbosity/etc.), an optional preference VECTOR in embedding space, and per-field
confidence (how much we actually know). Persisted in the workspace; rebuilt/updated incrementally."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import home


def _persona_path() -> Path:
    return home() / "persona.json"


def _vec_path() -> Path:
    return home() / "persona_vec.npy"


@dataclass
class PersonaProfile:
    field_weights: dict[str, float] = field(default_factory=dict)   # field -> emphasis (normalized)
    preferences: dict[str, str] = field(default_factory=dict)       # explicit style prefs
    confidence: dict[str, float] = field(default_factory=dict)      # field -> 0..1 data density
    n_signals: int = 0
    has_pref_vector: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- field weights ----
    def top_fields(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.field_weights.items(), key=lambda t: -t[1])[:k]

    def bump_field(self, fld: str, amount: float = 0.15) -> None:
        """Nudge a field's emphasis up (engaged) or down (negative amount); clamp >=0, renormalize."""
        if not fld or fld == "general":
            return
        self.field_weights[fld] = max(0.0, self.field_weights.get(fld, 0.0) + amount)
        self._renorm()

    def _renorm(self) -> None:
        tot = sum(self.field_weights.values())
        if tot > 0:
            self.field_weights = {k: round(v / tot, 4) for k, v in self.field_weights.items()}

    # ---- preference vector (optional, embedding space) ----
    def load_vector(self):
        if not (self.has_pref_vector and _vec_path().exists()):
            return None
        try:
            import numpy as np
            return np.load(_vec_path())
        except Exception:
            return None

    def update_vector(self, new_vecs, lr: float = 0.2, away: bool = False) -> None:
        """EMA the preference centroid toward (or, if away=True, away from) engaged-item embeddings."""
        try:
            import numpy as np
            target = np.asarray(new_vecs, dtype=np.float32)
            if target.ndim == 2:
                target = target.mean(axis=0)
            n = np.linalg.norm(target)
            if n == 0:
                return
            target = target / n
            cur = self.load_vector()
            if cur is None:
                vec = -target if away else target
            else:
                vec = cur + (-lr if away else lr) * target       # attract or repel
                vec = vec / (np.linalg.norm(vec) or 1.0)
            np.save(_vec_path(), vec.astype(np.float32))
            self.has_pref_vector = True
        except Exception:
            pass

    def summary(self) -> str:
        tf = ", ".join(f"{k} ({v:.0%})" for k, v in self.top_fields(4)) or "not yet learned"
        prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items()) or "defaults"
        conf = "low" if self.n_signals < 5 else ("medium" if self.n_signals < 25 else "high")
        return f"focus: {tf} | preferences: {prefs} | signals: {self.n_signals} (confidence {conf})"


def build_from_corpus(store) -> PersonaProfile:
    """Initialize field weights + confidence from the user's ingested corpus (project + chunk density)."""
    p = PersonaProfile()
    counts: dict[str, int] = {}
    chunks: dict[str, int] = {}
    with store.cursor() as c:
        for r in c.execute(
            "SELECT p.field AS field, count(DISTINCT p.id) AS np, count(ch.id) AS nc "
            "FROM projects p LEFT JOIN files f ON f.project_id=p.id LEFT JOIN chunks ch ON ch.file_id=f.id "
            "GROUP BY p.field").fetchall():
            d = dict(r)
            fld = d["field"] or "general"
            if fld == "general":
                continue
            counts[fld] = d["np"] or 0
            chunks[fld] = d["nc"] or 0
    tot = sum(counts.values()) or 1
    p.field_weights = {k: round(v / tot, 4) for k, v in counts.items()}
    # confidence: saturating function of chunk density per field (more data -> more confident)
    for fld, nc in chunks.items():
        p.confidence[fld] = round(1 - math.exp(-nc / 500.0), 3)
    return p


def load_persona(store=None) -> PersonaProfile:
    """Load the saved persona; if none and a store is given, bootstrap from the corpus."""
    pp = _persona_path()
    if pp.exists():
        try:
            data = json.loads(pp.read_text(encoding="utf-8"))
            return PersonaProfile(**{k: v for k, v in data.items() if k in PersonaProfile().__dict__})
        except Exception:
            pass
    p = build_from_corpus(store) if store is not None else PersonaProfile()
    save_persona(p)
    return p


def save_persona(p: PersonaProfile) -> Path:
    home().mkdir(parents=True, exist_ok=True)
    _persona_path().write_text(json.dumps(p.to_dict(), indent=2), encoding="utf-8")
    return _persona_path()
