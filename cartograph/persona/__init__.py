"""Persona layer — models the user as vectors + weights over the graph, learns what they respond to,
and steers any agent's context to fit them. The graph answers "what do I know"; the persona answers
"what does THIS user want, in THIS field, right now" and adapts as signals accumulate.

Concretely (the Hilbert-space metaphor, made buildable):
  - the embedding space is the high-dim space; the user is a set of vectors + field weights in it
  - feedback nudges those vectors (online EMA) toward what the user engaged with
  - retrieval is re-ranked by alignment to the persona; a model-agnostic "steering brief" is emitted
  - everything degrades gracefully: field-weight personalization needs zero ML; the preference VECTOR
    is an optional enhancement when semantic embeddings are installed
Honest framing + foundations: see docs/PERSONA.md. No personal data ships in this package.
"""
from .profile import PersonaProfile, load_persona
from .signals import record_feedback
from .steer import build_brief, personalized_retrieve

__all__ = ["PersonaProfile", "load_persona", "record_feedback", "build_brief", "personalized_retrieve"]
