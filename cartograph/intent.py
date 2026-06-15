"""Prompt-intent classifier — detects WHAT KIND of help the user wants (not which field), so the
steering brief tailors output shape to the use case: a debug ask, new code, a review, generative
writing, an explanation, or a general question. This is what makes Cartograph agnostic across the
user's whole range — 'from general questions to coding to field-specific generation'.

Zero-dependency, ordered-regex baseline (interpretable + always on). Each intent carries output
guidance an agent should follow; field-specific grounding is added separately by the router + persona.
"""
from __future__ import annotations

import re

# (intent, pattern, guidance) — ORDER matters: more specific intents first.
_INTENTS: list[tuple[str, re.Pattern, str]] = [
    ("debug", re.compile(r"\b(error|bug|broken|crash|traceback|exception|stack ?trace|fails?|failing|"
                         r"doesn'?t work|not working|why .* (fail|break|error))\b", re.I),
     "Debug: reproduce it, isolate the root cause (don't guess), propose the minimal fix, and add a "
     "regression test. Reference the user's own code patterns in the context below."),
    ("code", re.compile(r"\b(implement|write (a|some|the)?\s*(function|class|method|script|module|test|api|"
                        r"endpoint)|add (a|support)|build (a|the)|create (a|the)|refactor|migrate|port)\b", re.I),
     "Produce runnable code in the user's stack and conventions (mirror the context below); handle errors "
     "and edge cases; keep it minimal and tested. Prefer their existing utilities over new dependencies."),
    ("review", re.compile(r"\b(review|critique|improve|optimi[sz]e|make .* (better|faster|cleaner)|"
                          r"is this (good|correct|right)|code ?review|feedback on)\b", re.I),
     "Review against the field's bar: give specific, prioritized changes (most impactful first) with the "
     "reasoning; cite the user's own patterns. Be honest about trade-offs; don't rubber-stamp."),
    ("generate", re.compile(r"\b(write (an?|the)?\s*(email|post|blog|essay|readme|doc|report|summary|"
                            r"caption|copy|message|tweet)|draft|compose|summari[sz]e|rewrite|brainstorm)\b", re.I),
     "Match the user's voice and preferred format; lead with the point; keep it tight unless they ask for "
     "length. Use their domain vocabulary from the context below."),
    ("explain", re.compile(r"\b(explain|what is|what are|how does|how do|why (is|does|do)|difference between|"
                           r"understand|teach me|eli5|walk me through|clarify)\b", re.I),
     "Explain clearly in the user's domain vocabulary; concise first, depth on request; concrete examples "
     "grounded in their work where possible."),
]
_DEFAULT = ("general", "Answer directly and concisely in the user's domain vocabulary; ground in the "
            "context below; ask only if genuinely blocked.")


def classify(prompt: str) -> dict:
    """Return {intent, guidance} for a prompt. Use case, not field."""
    for name, pat, guidance in _INTENTS:
        if pat.search(prompt or ""):
            return {"intent": name, "guidance": guidance}
    return {"intent": _DEFAULT[0], "guidance": _DEFAULT[1]}
