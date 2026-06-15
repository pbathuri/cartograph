# The persona layer — steering your agents to you

> *"Assign vectors to all the things the user relates to; manipulate them, accurately or approximately
> by the probability of the data present, toward the response of best value for the user."*

This is the buildable form of that idea. Cartograph's graph answers **"what do I know?"** The persona
layer answers **"what does *this* user want — in *this* field, *right now* — and how should the answer
be shaped?"** It then emits a compact **steering brief** any model (Claude, Cursor, ChatGPT, Gemini)
can prepend, so outputs adapt to you and keep adapting as you use them.

## The Hilbert-space metaphor, made concrete
| Metaphor | Engineering |
|---|---|
| high-dimensional space of "everything you relate to" | the embedding space (R^d) over your ingested chunks |
| your persona as vectors in that space | per-field **weights** + an optional **preference vector** (centroid of what you engage with) |
| many complex subspaces | **one preference vector per field** (e.g. an `ml_experiment` direction *and* a `web_frontend` direction), + a global fallback — each item is scored in ITS OWN subspace |
| "direction vectors that manipulate the graph" | retrieval **re-ranking** = base rank blended with per-subspace persona alignment |
| "accurately or approximately by probability of data present" | a **confidence** term (data density per field + #signals) scales how hard we steer |
| "best value for the user" | the **steering brief**: persona + field + their patterns + output guidance |
| "adapts further with time" | **online EMA updates** to weights + preference vector from feedback |

So a vector that's well-supported by your data moves the answer a lot; a sparsely-supported one barely
nudges it. Low confidence → conservative steering; high confidence → strong steering. Honest by design.

## How it learns (the loop)
1. **Bootstrap** from your corpus: field weights + confidence from project/chunk density (`carto persona`).
2. **Signal**: when a result helps, record it (`carto feedback --liked <project>`, or your agent calls
   `personalize` then logs what it used). Field weights move via EMA; the preference vector moves toward
   the embeddings of what you engaged with.
3. **Steer**: `personalize(prompt)` retrieves, re-ranks by persona alignment, and returns the brief.
4. **Adapt**: every signal sharpens it. Early on it leans on declared field + corpus; over time it leans
   on your revealed preferences.

## Where this connects to the literature (honestly)
These are *related ideas that inform the design* — Cartograph implements lightweight, local,
inspectable versions, not the papers themselves:
- **Representation / steering / "persona" vectors** (activation-steering and representation-engineering
  work, incl. recent Anthropic interpretability on persona/feature directions): the idea that a
  direction in representation space corresponds to a behavior/trait you can lean toward. Cartograph
  steers *retrieval + context* (model-agnostic, no model internals needed), not activations — so it
  works with closed models you can't open.
- **Preference learning / RLHF** (Bradley–Terry-style preference modeling): we learn from "this was
  useful" signals, but as an online re-ranker over *your* data rather than fine-tuning a model.
- **Online learning / EMA / bandits**: incremental updates so the persona adapts without retraining.
- **Vector-space user modeling & collaborative filtering**: representing a user as vectors and scoring
  items by alignment.
- **Reciprocal Rank Fusion**: how semantic + keyword (+ persona) signals are combined without fragile
  score normalization.

If you want exact references, search: *representation engineering, activation steering, persona
vectors, RLHF preference models, reciprocal rank fusion, online learning to rank.*

## Connecting it to every model (what's real today)
- **Claude Code / Cursor / any MCP client** — first-class, now: add the MCP server and have the agent
  call the `personalize` tool at the start of a turn (put that instruction in its system prompt). Real,
  automatic, local.
- **Web GenAI (ChatGPT, Gemini, claude.ai)** — there is no universal "inject before every prompt" hook
  these vendors expose. The honest pattern: run `carto serve` (local HTTP `/personalize`), and use a
  small **browser userscript** that, on submit, fetches the brief and prepends it to your message.
  That userscript is the only per-vendor piece; the brain is shared. (Stub + guide: `docs/BROWSER.md`.)
- **Sync across tools**: they all read/write the *same* local persona + graph, so a preference learned
  in Cursor shapes your next ChatGPT answer. The "spiderweb in sync" is one shared local state, not N
  copies.

## Honest limits
- It steers **context and instructions**, not model weights or activations — so its power is bounded by
  how much the model respects provided context (high for modern models, not absolute).
- Persona is only as good as your signals + ingested data; new users get corpus/field-based steering,
  which sharpens with feedback.
- No personal data ships in this package; your persona lives only in your local workspace.
- This is a heuristic, inspectable system — every weight and the preference vector are on disk and
  explainable, by design. It is not a claim of modeling "emotion" or a person's interior; it models
  *revealed preferences over your own work and fields*.

## Why per-field subspaces (the value)
A single global preference vector blurs everything you like into one direction — so a strong signal in
your dominant field drags retrieval in *unrelated* fields toward it. Per-field vectors fix that: each
candidate snippet is scored against the preference direction **of its own field**, so "I like terse,
typed APIs" (library) and "I like accessible, state-first UIs" (web) coexist without bleeding into each
other. It's the honest realization of "many complex spaces, each with its own direction vector," and it
keeps steering sharp as your work spans more fields. Stored in `persona_vecs.npz` (one array per field +
`_global`), all inspectable.

## Built today vs. roadmap (honest)
**Built + tested now:** field weights from corpus density · **per-field preference subspaces** (one EMA
direction per field, attract *and* repel) + a global fallback · **recency decay** (old emphasis fades so
the persona tracks "now") · per-field confidence (data-density scaled steering) · subspace-aware
re-ranking · **learned steering strength (α)** — *learning-to-rank from the `record_use` log* · the
model-agnostic steering brief · explicit output-tuning preferences · MCP `personalize` + `record_use`
(implicit loop) · CLI + HTTP + browser-userscript surfaces · `carto demo`.

### How α self-tunes (learning-to-rank)
The blend `final = (1-α)·base_rank + α·persona_alignment` no longer uses a fixed α. Each feedback
signal asks: *did the persona's emphasis predict what the user found useful?* If you liked something in a
field the persona already favored (a **hit**), α rises — trust the persona more. If you valued something
**off**-persona, or disliked an on-persona item (a **miss**/over-steer), α falls. It's a bounded online
update (α ∈ [0.1, 0.7]) straight off your own accept/reject history — so steering strength is itself
learned from you, not hand-set. Persisted in `persona.json` (`learned_alpha`), shown in `carto persona`.

### The trained in-between-layer models
Cartograph now sits between your prompt and the model as a stack of small, **local, inspectable models
trained on your own data** — no fixed heuristics where learning is possible:
1. **Field router** (`carto train`) — one centroid embedding *per field* from your corpus; routes a
   prompt to its field by nearest centroid. Agnostic to *any* field you have, not a keyword list.
2. **Prompt-intent classifier** — detects the *use case* (debug / code / review / generate / explain)
   and shapes output guidance accordingly.
3. **Per-field preference subspaces** + **learned α** (above).
4. **Learned reranker** (`carto train`, once feedback exists) — a NumPy logistic model over 5 features
   (base rank, field weight, preference-vector cosine, field-match, **project affinity** from your
   `record_use` log). It learns *which signals predict what you found useful* and replaces the
   hand-tuned blend when trained; falls back to the heuristic otherwise. Honest: `fit_accuracy` is on
   your own history (it's a personal recommender — `proj_affinity` intentionally encodes standing
   preferences), not a held-out generalization claim. Stale models (feature-schema change) are ignored,
   never crash serving.

So the layer is **agnostic on three axes** — field (router), use-case (intent), and your revealed
preferences (subspaces + reranker) — and each axis is learned from *you*.

### The real-time vision layer (`carto watch`) — one core builder of data
The graph stops being a one-time ingest and becomes a **living stream**. A background loop periodically
screenshots, and a deliberately **light "model in between"** — a perceptual-hash + text-similarity
**novelty gate (a similarity cache)** — decides whether the frame is worth processing. Genuinely-new
screens are OCR'd, **redacted**, classified with the *same* router + intent models, and ingested as
graph chunks. Because they land in the same store, that stream flows into retrieval, the persona (field
weights gently drift toward what you actually do), and the steering brief — automatically. It's the
"one core builder of data": the graph grows from your real activity as time goes forward.

Why a cheap gate and not a heavy CV model? A background CPU loop's value is the **OCR text + dedup**, not
pixels. A CNN per tick would burn compute for almost no marginal signal; the reasoned choice is the
near-free gate that skips ~all redundant frames and routes the rest into the models we already trained.

**Privacy is the design, not a footnote:** local-only; **dry-run by default** (nothing stored until
`--apply`); **sensitive windows** (banking / password managers / auth) are never even OCR'd; **secrets
and PII are redacted** before anything is written; a `vision.paused` file is a one-touch kill-switch.

**Honest boundary on "optimize the output before the user sees it":** there is **no hook into a closed
model's (ChatGPT/Gemini) generation**. What's real is that the live screen context now rides in the
steering brief (`current_activity`), so the *injected context* reflects what you're doing right now —
the same model-agnostic mechanism as the rest of the persona layer, applied to a live signal.

**Natural next steps (not yet built):**
- **Cross-tool event stream** — a tiny local daemon that all surfaces write to, for true real-time sync.
- **Contextual α / per-field reranker** — separate learned weights per field/prompt-type.
- **Held-out reranker eval** — time-split the feedback log to report true generalization, not just fit.
- **Vision → session segmentation** — cluster the activity stream into tasks/sessions and summarize them
  into higher-order graph nodes (so "what was I doing yesterday" becomes a first-class query).

These are deliberately staged: the current version is simple, inspectable, and useful on day one;
the roadmap adds power without giving up that transparency.
