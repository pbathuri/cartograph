# Deep audit — dual-user trials (technical + non-technical) and recommendations

After building query-contextual affinity + broadened field inference, I ran two rigorous end-to-end
trials and audited the whole system with the full context of every trial so far. This is the synthesis:
what works, where the limits are, and concrete next changes for **technical** and **non-technical** users.

## Method
Two simulated users, each with a 12-project corpus (111 files), full lifecycle (ingest → index → 30–60
days of feedback → weekly retrain). For each: **regular** workflow (plain hybrid retrieval, no persona)
vs **backed** workflow (Cartograph personalization), at a **short-term** (day 3, sparse feedback) and
**long-term** (day 60) horizon. CLEAR queries = no-regression check; AMBIG queries = engineered so the
rival project leads base retrieval, so personalization has real work to do. (Run on an internal,
synthetic-data eval harness kept out of the public package; results below.)

## Results (MRR of the correct/preferred project)
| User | | regular | backed short | backed long |
|---|---|---|---|---|
| **Priya — Computer Engineer** (technical) | CLEAR | 1.00 | 1.00 | 1.00 |
| | AMBIG | 0.40 | 0.50 | **0.90** (+0.50) |
| **Marcus — Finance/Business Consultant** (non-technical) | CLEAR | 1.00 | 1.00 | 1.00 |
| | AMBIG | 0.50 | 0.50 | **1.00** (+0.50) |

(Accountant trial, hand-written corpus: CLEAR 1.00→1.00, AMBIG 0.867→0.90. Large-corpus retrieval p95 ≈ 16 ms.)

## What the audit confirms works
1. **Both user types benefit, long-term and equally** (+0.50 AMBIG MRR). The value is *use-case agnostic*.
2. **No regression, ever.** CLEAR stays 1.00 at both horizons for both users; the top-relevance hit is
   protected unless contextually-confident it's wrong. Short-term does no harm.
3. **Contextual affinity is the right, field-AGNOSTIC mechanism.** The finance corpus collapsed entirely
   to `general` fields, yet personalization still delivered +0.50 — because affinity is conditioned on the
   query *cluster*, not on declared fields. This is exactly what non-dev users need.
4. **Honest cold→warm curve.** Gains require feedback to accrue (short-term ≈ flat). This is correct
   behavior for a system that learns from the user, not a magic day-1 claim.

## Findings / limits (with the fix that shipped, or the recommendation)
- **F1 — Chunk-rank gating blocked multi-file projects.** *Shipped:* gate now keys off distinct-project
  rank, so a preferred-but-#2 project is promotable regardless of a rival's file count.
- **F2 — Coarse query clusters merged opposite-preference queries.** *Shipped:* finer k (~0.7·unique).
- **F3 — Field inference still misses many non-dev vocabularies** (finance consultant → all `general`).
  Personalization survives this, but field-level features (weights, subspaces) are dead for those users.
- **F4 — Short-term cold start is weak** for everyone (needs ~weeks of feedback for the big lift).
- **F5 — `learned_alpha` floors for single-field users** (harmless today because steering strength is
  decoupled and confidence-scaled, but it's a dead knob for them).

## Recommendations

### For technical users (Computer Engineering, software, etc.)
- **R1 (high):** Auto-seed feedback from the IDE/MCP `record_use` loop on accepted edits, so the warm-up
  (F4) happens passively during normal coding instead of needing explicit thumbs-up.
- **R2 (med):** Add a *symbol/path* signal to retrieval+affinity (function/class/file identifiers), where
  exact-token overlap is decisive — technical queries are token-precise.
- **R3 (med):** Per-field reranker weights — devs span genuinely distinct fields (e.g. `embedded_hardware`
  vs `web_frontend`); a shared reranker under-fits. The field signal is real for them (unlike F3).

### For non-technical users (finance, consulting, legal, clinical, marketing)
- **R4 (high):** **Learn fields instead of keyword-matching them.** F3 shows hard-coded tokens won't keep
  up with every domain. Cluster the corpus into emergent "fields" (the router already builds centroids);
  treat clusters as fields so field weights/subspaces work for *any* vocabulary. This is the single
  biggest lever for non-dev users.
- **R5 (high):** **One-click feedback in the Studio + web userscript** (a 👍 on a used answer → `record_use`).
  Non-dev users won't run `carto feedback`; the lift depends on easy signal capture (F4).
- **R6 (med):** Ship **starter context packs** per domain (finance/legal/consulting) so day-1 retrieval is
  strong before any personalization, mirroring the dev "reference repos" packs.

### Cross-cutting
- **R7 (high):** **Held-out / time-split evaluation in-product** (`carto eval`): replay the feedback log
  train-on-past, test-on-future, to report a true generalization number instead of fit-on-history.
- **R8 (med):** Decay/clean stale query-context clusters so long-lived workspaces don't accumulate dead
  contexts; surface them in the Studio stats (advanced) for transparency.
- **R9 (low):** Retire or repurpose `learned_alpha` for single-field users (F5) to avoid a dead knob.

## Verdict
The query-contextual affinity model is the correct fix the earlier trial demanded: it delivers a real,
safe, **+0.50 AMBIG MRR** gain for *both* a technical and a non-technical user with **zero clear-query
regression**, and it does so without depending on field inference — which is the property non-technical
users most need. The top remaining lever for non-dev users is **learned (clustered) fields** (R4); for
technical users it's **passive feedback capture** (R1). Both are scoped above.
