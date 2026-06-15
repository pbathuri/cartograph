# Scaling

Cartograph's defaults are tuned for "everything one person works with" — comfortably millions of chunks.

## Where the limits are
| Component | Comfortable | Beyond |
|---|---|---|
| SQLite + FTS5 | tens of millions of chunks | fine; it's just a bigger file |
| Brute-force semantic search | ~1–5M vectors (~150 ms/query) | swap in an ANN index (below) |
| Embedding throughput | GPU: ~hundreds–thousands chunks/s | batch + fp16 (already on); more GPUs |

## Faster keyword search
Already applied: `ORDER BY rank` uses FTS5's optimized top-k path (≈3× faster than `bm25()` ordering
on large corpora). No action needed.

## ANN for very large corpora (FAISS)
Past a few million vectors, replace the brute-force dot product in `embed.py:SemanticIndex` with FAISS:

```python
import faiss, numpy as np
vecs = np.load(index_dir()/"vectors.npy")            # float16 memmap
idx = faiss.IndexFlatIP(vecs.shape[1])               # or IndexHNSWFlat for sublinear
idx.add(vecs.astype("float32"))
faiss.write_index(idx, str(index_dir()/"faiss.idx"))
# query: idx.search(qv[None].astype("float32"), top_k)
```
`IndexHNSWFlat` gives sub-linear queries at high recall; `IndexIVFFlat` trades a tiny recall hit for
much less memory. Keep `chunk_ids.npy` / `project_ids.npy` for the id mapping.

## Keeping it fast
- Put `CARTOGRAPH_HOME` on an SSD/NVMe.
- Re-`carto index` after large ingests (it rebuilds in place).
- For multi-machine sharing, ship a **reference pack** (see REFERENCE_PACKS.md), not your live graph.
