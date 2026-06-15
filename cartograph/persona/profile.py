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


def _vecs_path() -> Path:
    # one preference vector PER field (the "many subspaces") + a "_global" fallback, in one npz.
    return home() / "persona_vecs.npz"

_GLOBAL = "_global"


@dataclass
class PersonaProfile:
    field_weights: dict[str, float] = field(default_factory=dict)   # field -> emphasis (normalized)
    preferences: dict[str, str] = field(default_factory=dict)       # explicit style prefs
    confidence: dict[str, float] = field(default_factory=dict)      # field -> 0..1 data density
    n_signals: int = 0
    has_pref_vector: bool = False
    learned_alpha: float = 0.35      # how hard to steer; LEARNED from whether the persona predicted hits

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def tune_alpha(self, predicted: bool, lr: float = 0.03) -> None:
        """Learning-to-rank: if the persona's emphasis matched what the user found useful, trust it more
        (raise alpha); if it missed (user valued something off-persona, or disliked an on-persona item),
        steer less. Bounded so it can never fully dominate or vanish."""
        self.learned_alpha = round(min(0.7, max(0.1, self.learned_alpha + (lr if predicted else -lr))), 4)

    # ---- field weights ----
    def top_fields(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.field_weights.items(), key=lambda t: -t[1])[:k]

    def decay(self, factor: float = 0.98) -> None:
        """Recency: gently shrink existing field emphasis so newer signals weigh more ('right now')."""
        if 0 < factor < 1 and self.field_weights:
            self.field_weights = {k: v * factor for k, v in self.field_weights.items()}

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

    # ---- preference vectors: one per field (subspaces) + a global fallback (embedding space) ----
    def _load_all(self) -> dict:
        if not _vecs_path().exists():
            return {}
        try:
            import numpy as np
            with np.load(_vecs_path()) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            return {}

    def _save_all(self, d: dict) -> None:
        try:
            import numpy as np
            np.savez(_vecs_path(), **d)
            self.has_pref_vector = bool(d)
            self.pref_fields = sorted(k for k in d if k != _GLOBAL)
        except Exception:
            pass

    def load_vector(self, field: str | None = None):
        """The preference vector for a field's subspace; falls back to the global vector."""
        d = self._load_all()
        if not d:
            return None
        if field and field in d:
            return d[field]
        return d.get(_GLOBAL)

    def update_vector(self, new_vecs, field: str | None = None, lr: float = 0.2, away: bool = False) -> None:
        """EMA the chosen field's subspace vector (and the global) toward/away from engaged embeddings."""
        try:
            import numpy as np
            target = np.asarray(new_vecs, dtype=np.float32)
            if target.ndim == 2:
                target = target.mean(axis=0)
            n = np.linalg.norm(target)
            if n == 0:
                return
            target = target / n
            d = self._load_all()

            def _ema(cur):
                if cur is None:
                    return -target if away else target
                v = cur + (-lr if away else lr) * target
                return v / (np.linalg.norm(v) or 1.0)

            keys = [_GLOBAL] + ([field] if field and field != "general" else [])
            for k in keys:
                d[k] = _ema(d.get(k)).astype(np.float32)
            self._save_all(d)
        except Exception:
            pass

    def summary(self) -> str:
        tf = ", ".join(f"{k} ({v:.0%})" for k, v in self.top_fields(4)) or "not yet learned"
        prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items()) or "defaults"
        conf = "low" if self.n_signals < 5 else ("medium" if self.n_signals < 25 else "high")
        subs = getattr(self, "pref_fields", []) or [k for k in self._load_all() if k != _GLOBAL]
        sub = f" | subspaces: {', '.join(subs)}" if subs else ""
        return (f"focus: {tf} | preferences: {prefs} | signals: {self.n_signals} (confidence {conf}) "
                f"| steer α={self.learned_alpha:.2f}{sub}")


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
