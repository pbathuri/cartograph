"""Retrieval modes with hard token budgets, chunk de-duplication, and the optional ANN accelerator."""
import json

import pytest

from cartograph import memory as M
from cartograph.config import Config, db_path
from cartograph.ingest import ingest_path
from cartograph.mcp_server import _handle
from cartograph.retrieve import _dedupe, context, tokens
from cartograph.storage import Store


@pytest.fixture
def graph(tmp_path):
    d = tmp_path / "ocular"
    d.mkdir()
    (d / ".git").mkdir()
    long = "\n".join(f"Timolol corneal permeability measured in Franz cells, run {i}, flux {i * 0.1:.1f}."
                     for i in range(200))
    (d / "notes.md").write_text(long)
    (d / "refs.md").write_text("Timolol transport: arXiv:2401.04444 and doi 10.1167/iovs.62.1.1\n")
    (d / "requirements.txt").write_text("pydantic\nnumpy\n")
    store = Store(db_path())
    ingest_path(d, store, Config())
    M.remember(store, "Timolol permeability runs use the Franz cell protocol v2", "decision", cwd=str(d))
    M.remember(store, "Rejected: Ussing chamber for timolol, too much edge damage", "rejected", cwd=str(d))
    return d, store


@pytest.mark.parametrize("budget", [200, 350, 600, 1000, 2500])
def test_mix_never_exceeds_budget(graph, budget):
    d, store = graph
    out = context("timolol permeability", store, Config(), mode="mix", max_tokens=budget, cwd=str(d))
    payload = {k: out[k] for k in ("memories", "facts", "chunks") if k in out}
    assert tokens(json.dumps(payload)) <= budget and out["tokens"] <= budget


def test_mix_has_memories_facts_and_cited_chunks(graph):
    d, store = graph
    out = context("timolol permeability arXiv 2401.04444", store, Config(), mode="mix", max_tokens=2500, cwd=str(d))
    assert {m["kind"] for m in out["memories"]} >= {"decision", "rejected"}
    assert any(f["entity"] == "paper:arxiv:2401.04444" for f in out["facts"])
    assert out["chunks"] and all(c["cite"] for c in out["chunks"])
    assert any(":L" in c["cite"] for c in out["chunks"])


def test_modes(graph):
    d, store = graph
    cfg = Config()
    g = context("pydantic", store, cfg, mode="graph")
    assert g["facts"][0]["entity"] == "package:pypi/pydantic" and "chunks" not in g
    p = context("timolol", store, cfg, mode="projects")
    assert p["projects"][0]["project"] == "ocular" and p["projects"][0]["best"]
    m = context("ussing chamber", store, cfg, mode="memory", cwd=str(d))
    assert [x["kind"] for x in m["memories"]] == ["rejected"] and "chunks" not in m
    with pytest.raises(ValueError):
        context("x", store, cfg, mode="everything")


def test_dedupe_drops_overlapping_windows():
    a = {"file_path": "f.md", "chunk_text": "alpha beta gamma delta epsilon zeta eta theta"}
    b = {"file_path": "f.md", "chunk_text": "gamma delta epsilon zeta eta theta iota kappa"}
    c = {"file_path": "g.md", "chunk_text": "gamma delta epsilon zeta eta theta iota kappa"}
    assert _dedupe([a, b, c]) == [a, c]


def test_mcp_retrieve_context_modes(graph):
    d, _ = graph
    call = lambda args: json.loads(json.loads(_handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",  # noqa: E731
        "params": {"name": "retrieve_context", "arguments": args}}))["result"]["content"][0]["text"])
    legacy = call({"query": "timolol"})
    assert set(legacy) == {"method", "chunks"} and legacy["chunks"][0]["cite"]
    mixed = call({"query": "timolol", "mode": "mix", "max_tokens": 500, "cwd": str(d)})
    assert mixed["tokens"] <= 500 and mixed["memories"]


# ---------------------------------------------------------------------------------------------- ANN
def test_ann_matches_exact_search(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from cartograph import ann, embed
    from cartograph.config import index_dir

    dim, n = 32, 4000
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(40, dim))
    X = centers[rng.integers(0, 40, n)] + 0.05 * rng.normal(size=(n, dim))
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float16)
    d = index_dir()
    d.mkdir(parents=True)
    np.save(d / "vectors.npy", X)
    np.save(d / "chunk_ids.npy", np.arange(1, n + 1, dtype=np.int64))
    np.save(d / "project_ids.npy", np.zeros(n, dtype=np.int64))
    info = ann.build(d, min_rows=1000)
    assert info["rows"] == n and (d / "ann" / "codes.npy").exists()

    class Q:
        def encode(self, texts, normalize_embeddings=True):
            i = int(texts[0].split()[-1])
            return np.asarray(X[i:i + 1], dtype=np.float32)

    monkeypatch.setattr(embed, "_model", lambda name: Q())
    idx = embed.SemanticIndex(Config())
    assert idx.mode == "ann"
    hits = 0
    for i in range(0, n, 97):
        got = idx.search_chunk_ids(f"query {i}", top_k=10)
        idx_ann, idx._ann_ok = idx.ann, False
        idx.ann = None
        exact = idx.search_chunk_ids(f"query {i}", top_k=10)
        idx.ann, idx._ann_ok = idx_ann, True
        hits += len(set(got) & set(exact))
        assert got[0] == i + 1                               # the query's own vector ranks first
    assert hits / (10 * len(range(0, n, 97))) >= 0.9


def test_ann_skips_small_indexes(tmp_path):
    np = pytest.importorskip("numpy")
    from cartograph import ann
    from cartograph.config import index_dir
    d = index_dir()
    d.mkdir(parents=True)
    np.save(d / "vectors.npy", np.ones((10, 8), dtype=np.float16))
    assert "skipped" in ann.build(d)
