"""Real-time vision layer — the continuous, privacy-first data source.

Periodically capture the screen -> a cheap novelty gate decides if it's worth processing (similarity
cache) -> OCR only the genuinely-new screens -> redact secrets/PII -> classify field + intent with the
SAME trained models the rest of Cartograph uses -> ingest as graph chunks. That stream becomes "one core
builder of data": it flows into retrieval, the persona, and the steering brief automatically, so the
graph grows from what you actually do, over time.

Design stance (honest):
  * Opt-in, local-only, dry-run by default. Sensitive windows are skipped; secrets are redacted.
  * The "ML model in between" is deliberately LIGHT — a perceptual-hash + text-similarity novelty gate,
    not a heavy CNN. A background CPU loop's value is the OCR text + dedup, so the reasoned choice is the
    cheap gate (skip ~all redundant frames) feeding the models we already trained (router/intent/persona).
  * There is NO hook into a closed model's (ChatGPT/Gemini) internal generation. The real mechanism is
    enriching the context injected BEFORE the prompt (the steering brief now carries live screen context).
"""
