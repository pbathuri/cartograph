"""Index builds survive being killed: they resume from the rows already embedded, embed the user's own
roots first, and are searchable while still building. A fake encoder keeps these tests model-free."""
import hashlib
import os
import time

import pytest

np = pytest.importorskip("numpy")

from cartograph import embed  # noqa: E402
from cartograph.config import Config, db_path, index_dir  # noqa: E402
from cartograph.storage import Store  # noqa: E402

DIM = 16


class FakeModel:
    """Deterministic unit vectors from text; counts how many texts it was asked to embed."""

    def __init__(self):
        self.calls = 0

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        self.calls += len(texts)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h[:DIM], dtype=np.uint8).astype(np.float32) + 1.0
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class Boom(Exception):
    pass


@pytest.fixture()
def fake(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(embed, "_model", lambda name: m)
    return m


def _corpus(tmp_path, roots):
    store = Store(db_path())
    for r, name, n in roots:
        pid = store.upsert_project(name, str(tmp_path / r / name), "general")
        fid = store.upsert_file(pid, f"{name}.md", ".md", name)
        store.add_chunks(fid, [f"{name} chunk {i}" for i in range(n)])
    return store


def test_killed_build_resumes_without_reembedding(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 5), ("b", "beta", 5)])
    cfg = Config(roots=[str(tmp_path / "a"), str(tmp_path / "b")])
    seen = []

    def die_after_first_batch(msg):
        seen.append(msg)
        raise Boom

    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=4, progress=die_after_first_batch, progress_every=1)
    building = index_dir().parent / "index.building"
    assert (building / "chunk_ids.npy").exists()           # partial build is self-describing
    fake.calls = 0
    res = embed.build_index(store, cfg, batch=4)
    assert res["resumed"] == 4 and res["embedded"] == 6
    assert fake.calls == 6                                 # only the missing rows were embedded
    fresh = embed.build_index(store, cfg, batch=4, full=True)
    assert fresh["embedded"] == 10
    assert not building.exists()


def test_resumed_vectors_match_a_clean_build(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 3), ("b", "beta", 4)])
    cfg = Config(roots=[str(tmp_path / "a"), str(tmp_path / "b")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=2, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    embed.build_index(store, cfg, batch=2)
    resumed = np.load(index_dir() / "vectors.npy").copy()
    embed.build_index(store, cfg, batch=2, full=True)
    assert np.array_equal(resumed, np.load(index_dir() / "vectors.npy"))


def test_first_root_is_embedded_first(tmp_path, fake):
    store = _corpus(tmp_path, [("refs", "cloned", 4), ("mine", "thesis", 4)])
    cfg = Config(roots=[str(tmp_path / "mine"), str(tmp_path / "refs")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=4, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    building = index_dir().parent / "index.building"
    vecs = np.load(building / "vectors.npy")
    pids = np.load(building / "project_ids.npy")
    done = np.any(vecs != 0, axis=1)
    thesis = store.project_id_by_name("thesis")
    assert set(pids[done].tolist()) == {thesis}            # the user's own root came first


def test_partial_build_is_searchable(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 4), ("b", "beta", 4)])
    cfg = Config(roots=[str(tmp_path / "a"), str(tmp_path / "b")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=4, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    assert embed.SemanticIndex.exists()
    idx = embed.SemanticIndex(cfg)
    assert idx.building
    alpha_ids = {r["chunk_id"] for r in store.search_chunks('"alpha"', limit=10)}
    hits = idx.search_chunk_ids("search_query: alpha chunk 1", top_k=4)
    assert hits and set(hits) <= alpha_ids                  # only rows that are embedded come back


def test_index_without_chunk_ids_is_not_usable(tmp_path):
    d = index_dir()
    d.mkdir(parents=True)
    np.save(d / "vectors.npy", np.zeros((3, DIM), dtype=np.float16))
    assert not embed.SemanticIndex.exists()


def test_legacy_partial_build_resumes_only_if_graph_unchanged(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 4)])
    cfg = Config(roots=[str(tmp_path / "a")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=2, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    building = index_dir().parent / "index.building"
    for n in ("chunk_ids.npy", "project_ids.npy", "progress.json"):   # what an older version left behind
        (building / n).unlink(missing_ok=True)
    started = embed._build_started(building)
    old = started - 3600
    for p in (db_path(), db_path().with_name(db_path().name + "-wal")):
        if p.exists():
            os.utime(p, (old, old))
    fake.calls = 0
    assert embed.build_index(store, cfg, batch=2)["resumed"] == 2
    assert fake.calls == 2

    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=2, full=True, progress=lambda m: (_ for _ in ()).throw(Boom()),
                          progress_every=1)
    for n in ("chunk_ids.npy", "project_ids.npy", "progress.json"):
        (building / n).unlink(missing_ok=True)
    newer = time.time() + 5
    os.utime(db_path(), (newer, newer))                    # graph written after the build began
    res = embed.build_index(store, cfg, batch=2)
    assert res["resumed"] == 0 and res["reused"] == 4      # partial build refused; prior index reused


def test_graph_change_mid_build_carries_finished_rows_over(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 6), ("b", "beta", 4)])
    cfg = Config(roots=[str(tmp_path / "a"), str(tmp_path / "b")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=4, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    pid = store.upsert_project("gamma", str(tmp_path / "a" / "gamma"), "general")   # a nightly ingest lands
    store.add_chunks(store.upsert_file(pid, "g.md", ".md", "gamma"), [f"gamma chunk {i}" for i in range(3)])
    fake.calls = 0
    res = embed.build_index(store, cfg, batch=4)
    assert res["carried"] == 4 and res["vectors"] == 13
    assert fake.calls == 9                                 # the 4 finished rows were not re-embedded
    assert not (index_dir().parent / "index.carried").exists()
    carried = np.load(index_dir() / "vectors.npy").copy()
    embed.build_index(store, cfg, batch=4, full=True)
    assert np.array_equal(carried, np.load(index_dir() / "vectors.npy"))


def test_second_builder_leaves_a_live_build_alone(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 6)])
    cfg = Config(roots=[str(tmp_path / "a")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=2, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    building = index_dir().parent / "index.building"
    before = np.load(building / "vectors.npy").copy()
    embed._lock_path().write_text(str(os.getppid()))      # another live process holds the build
    with pytest.raises(embed.BuildRunning):
        embed.build_index(store, cfg, batch=2, full=True)
    assert np.array_equal(before, np.load(building / "vectors.npy"))


def test_lock_of_a_dead_builder_is_taken_over(tmp_path, fake):
    import subprocess
    import sys
    store = _corpus(tmp_path, [("a", "alpha", 3)])
    cfg = Config(roots=[str(tmp_path / "a")])
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    embed._lock_path().write_text(str(dead.pid))
    assert embed.build_index(store, cfg, batch=2)["vectors"] == 3
    assert not embed._lock_path().exists()


def test_nightly_improve_skips_the_index_while_a_build_runs(tmp_path, fake, monkeypatch):
    from cartograph import improve
    _corpus(tmp_path, [("a", "alpha", 3)])
    monkeypatch.setattr(embed, "available", lambda: True)
    embed._lock_path().write_text(str(os.getppid()))
    rep = improve.run(do_ingest=False, do_scout=False, do_tune=False, max_queries=0)
    step = rep["steps"]["index"]
    assert step["ok"] and "already running" in step["result"]["skipped"]
    assert (improve._reports_dir() / "latest.json").exists()


def test_progress_file_reports_counts(tmp_path, fake):
    store = _corpus(tmp_path, [("a", "alpha", 6)])
    cfg = Config(roots=[str(tmp_path / "a")])
    with pytest.raises(Boom):
        embed.build_index(store, cfg, batch=2, progress=lambda m: (_ for _ in ()).throw(Boom()), progress_every=1)
    st = embed.index_status()
    assert st["state"] == "building" and st["done"] == 2 and st["total"] == 6
