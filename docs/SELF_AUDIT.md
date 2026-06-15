# Self-audit — is the current build optimal, and does it generalize?

I stress-tested my own recent design choices: an **ablation** of each personalization component, a
**held-out generalization** test (the shortcoming I'd flagged: gains measured on the same queries used
for feedback), and a **hyperparameter sweep**. Run on the *finance consultant* world — the hardest case,
where field inference fully collapses to `general`, so only the query-contextual mechanism can help.
Harness: `test_v1/ablation.py` (steering knobs exposed via `personalized_retrieve(..., tuning=...)`).

## Held-out generalization + component ablation
Trained on `AMBIG_TRAIN` queries; `AMBIG_heldout` are **unseen paraphrases** (same intent/preferred
project, different wording — never in the feedback log).

| config | CLEAR | AMBIG (seen) | AMBIG (held-out) |
|---|---|---|---|
| regular (no persona) | 1.000 | 0.500 | 0.700 |
| **contextual (PRODUCTION)** | **1.000** | **1.000** | **1.000** |
| global affinity (no context) | 1.000 | 0.500 | 0.700 |
| no relevance gate | 1.000 | 1.000 | 1.000 |
| no top-hit protection | 1.000 | 1.000 | 1.000 |

## Hyperparameter sweep (contextual; CLEAR / AMBIG held-out)
`gate_div ∈ {1,2,3}` → all 1.00 / 1.00.  `dislike_thr ∈ {-0.2,-0.4,-0.6}` → all 1.00 / 1.00.

## Conclusions
1. **It GENERALIZES.** Contextual affinity scores **1.000 on unseen paraphrases** vs 0.700 regular — it is
   not memorising exact queries. Paraphrases embed near the trained query-cluster and inherit the
   preference. This was my biggest open worry; it is now answered with data.
2. **Contextual affinity is NECESSARY, not cosmetic.** Global affinity == regular (no benefit), because
   when preference is query-contextual each project is preferred in one query and a rival in another, so a
   single global number cancels to ~0. Only per-cluster affinity carries signal.
3. **Gate + top-hit protection are safe insurance.** They don't differentiate *here* (contextual affinity
   is well-behaved, nothing to protect against), but they were load-bearing in the accountant trial, where
   concentrated feedback created a strong GLOBAL affinity that, ungated/unprotected, regressed clear-query
   MRR 1.00 → 0.44. Keeping them costs nothing measurable and prevents that failure mode. Verified safe.
4. **Robust, not knife's-edge.** The system is insensitive to the gate divisor and dislike threshold over
   a wide range, so the chosen values aren't fragile.

**Verdict: the current build is Pareto-optimal among the configurations tested** — maximum lift
(seen + held-out), zero clear-query regression, and robust to its knobs. No simpler config does better;
the extra guardrails are justified by a *different* scenario rather than this one.

## Honest scope — what I did NOT yet sweep (perspectives still open)
- **Lower-level knobs:** RRF fusion `k` (60), reranker GD iterations/L2, the embedding model itself,
  k-means init/iterations. These predate this work and weren't ablated here; they're candidates for a
  future `carto eval` sweep.
- **Single hard scenario for the guardrails:** gate/protection necessity rests on the accountant trial,
  not this finance ablation. A unified multi-scenario eval would make that crisper.
- **The real next lever is architectural, not a tuning value:** *learned/clustered fields* (audit R4) —
  for non-dev users whose fields collapse to `general`, giving field-level features a learned substrate.
  That is a capability addition, and the ablation shows it's where remaining headroom is, not in retuning
  the contextual stack (which is already saturating these scenarios).
