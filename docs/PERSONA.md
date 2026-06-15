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
| many complex subspaces | per-field subsets; the prompt selects which subspace dominates |
| "direction vectors that manipulate the graph" | retrieval **re-ranking** = base rank blended with persona alignment |
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
