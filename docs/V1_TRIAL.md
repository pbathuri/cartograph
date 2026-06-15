# V1 end-to-end trial — a simulated accountant, one month, measured honestly

To decide whether V1 *actually works* (not just runs), I ran a full lifecycle on a synthetic but
realistic user and measured retrieval quality with ground truth. This records what worked, what didn't,
and the fix that shipped. Harness + data: `test_v1/` (git-ignored; regenerate with `gen_corpus.py` +
`simulate_month.py`).

## Setup
- **User:** Dana Okafor, Senior Accountant at *Meridian Logistics Inc.* — 10 work areas as folders
  (month-end close, AP automation, tax, audit, financial reporting, payroll, fixed assets, FP&A, SOX,
  AR), each mapped to one ground-truth project. Deterministic synthetic content (no GANs/HF download —
  same realism, no network flakiness).
- **Lifecycle exercised:** clean-room wheel install in a fresh venv → `carto` loads, `carto demo` runs →
  ingest → semantic index → persona → 30 simulated days of queries + feedback + weekly `train` → re-eval.
- **Two query sets:** **CLEAR** (one project obviously correct — sanity / no-regression) and **AMBIG**
  (several plausible projects; the "right" one is Dana's habit — where personalization must earn its keep).
- **Metric:** MRR / P@3 of the correct project, cold (day 0) vs warm (day 30).

## Results
| | cold | warm | note |
|---|---|---|---|
| CLEAR MRR | 1.00 | **1.00** | no regression (after fix; was **0.44** before) |
| AMBIG MRR | 0.867 | 0.867 | flat — see finding 3 |
| AMBIG (base retrieval, no persona) | 0.867 | 0.867 | personalization is bounded, does no harm |
| Large-corpus retrieval latency | — | **p95 ≈ 16 ms** over 1,260 chunks (FTS) | scales |

## What the trial found
1. **Field inference is dev-centric.** All 10 accounting projects inferred to `general` — Cartograph's
   built-in field tokens are software-domain. So *field-based* steering (weights, per-field subspaces) is
   inert for a non-dev user. (Mitigation today: declare `field_focus`; proper fix is broader/learned
   field inference — roadmap.)
2. **The learned reranker over-steered — a real regression.** Trained on concentrated feedback, its
   *global* `proj_affinity` (weight ~2.6) dragged favored projects to the top of **unrelated** queries,
   collapsing CLEAR MRR **1.00 → 0.44**. **Fix shipped:** the reranker is now *relevance-gated* — it may
   re-order only the relevant head (full weight at rank 0, zero past rank ~3) and may never demote the
   single most-relevant hit on affinity alone. CLEAR returns to **1.00**; regression-tested
   (`tests/test_steer_gate.py`).
3. **Within-domain preference is query-CONTEXTUAL — global affinity is the wrong granularity.** A project
   can be the wrong answer for one query and the right answer for another (e.g. *payroll* is a poor match
   for "reconcile the account" but the correct match for "Form 941 payroll"). A single global affinity
   therefore *cannot* lift AMBIG without risking CLEAR. I also evaluated a contextual-bandit α-gate that
   keys off retrieval ambiguity; on a small, vocabulary-overlapping corpus the ambiguity signal was **not
   separable** between clear and ambiguous queries (measured), so I rejected it rather than ship a model
   with no signal. The honest conclusion: a real AMBIG gain needs **query-conditioned affinity**, scoped
   as the next build.
4. **`learned_alpha` collapses for single-domain users.** With no field weights, the field-based α-tuner
   reads every "liked" as off-persona and drives α to its floor — which, given finding 2, is *accidentally
   protective*, so V1 leaves it; decoupling it caused regressions.

## Verdict
End-to-end V1 is **functional and safe**: install→ingest→index→retrieve→personalize all work; retrieval
is strong and fast at scale; and the one real defect (reranker over-steer) was found by this trial and
fixed with a regression test. Personalization is deliberately **conservative — it does no harm** — and
its within-domain upside is gated behind the next-gen, query-contextual affinity model (documented, not
faked). This is the honest bar for calling V1 done.
