"""Stale and duplicate rows never reach the agent: pruning on re-ingest, identical copies collapsed,
off-topic session logs kept out of mixed answers, keyword search that ignores question words."""
from cartograph import memory as M
from cartograph.config import Config, db_path
from cartograph.ingest import ingest_path
from cartograph.retrieve import context, retrieve
from cartograph.storage import Store


def _repo(d, name, text):
    r = d / name
    r.mkdir(parents=True)
    (r / ".git").mkdir()
    (r / "main.py").write_text(text)
    return r


def _paths(store):
    with store.cursor() as c:
        return [r[0] for r in c.execute("SELECT path FROM files")]


def test_deleted_file_is_dropped_on_reingest(tmp_path):
    d = tmp_path / "lab"
    d.mkdir()
    (d / ".git").mkdir()
    (d / "keep.md").write_text("Franz cell permeability protocol, version two.")
    (d / "gone.md").write_text("Ussing chamber notes that were later deleted.")
    store = Store(db_path())
    ingest_path(d, store, Config())
    (d / "gone.md").unlink()
    st = ingest_path(d, store, Config())
    assert st.files_pruned == 1
    assert [p.endswith("keep.md") for p in _paths(store)] == [True]
    assert not retrieve("Ussing chamber", store, Config()).chunks


def test_whole_folder_project_split_into_repos_keeps_one_copy_of_each_file(tmp_path):
    refs = tmp_path / "References"
    _repo(refs, "alpha", "def alpha_kernel():\n    return 'tiled matmul'\n")
    (refs / "index.md").write_text("reference clones")
    store = Store(db_path())
    ingest_path(refs, store, Config())                     # one repo inside: ingested as one blob
    _repo(refs, "beta", "def beta_kernel():\n    return 'groupby'\n")
    st = ingest_path(refs, store, Config())                # now a container of two repos
    paths = _paths(store)
    assert len(paths) == len(set(paths)) == 3              # alpha/main.py, beta/main.py, index.md
    assert st.files_pruned == 1                            # alpha/main.py left the blob project
    hits = retrieve("alpha_kernel tiled matmul", store, Config()).chunks
    assert len(hits) == 1 and hits[0]["project_name"] == "alpha"


def test_identical_copies_of_a_file_return_one_passage(tmp_path):
    code = "def volatility_target(returns, annual=0.25):\n    return annual / returns.std()\n"
    _repo(tmp_path / "code", "trading", code)
    _repo(tmp_path / "code", "trading-copy", code)
    store = Store(db_path())
    ingest_path(tmp_path / "code", store, Config())
    hits = retrieve("volatility_target annual", store, Config()).chunks
    assert len(hits) == 1


def test_mix_keeps_off_topic_session_logs_out_but_curated_memories_in(tmp_path):
    d = tmp_path / "ocular"
    d.mkdir()
    (d / ".git").mkdir()
    (d / "notes.md").write_text("Timolol corneal permeability measured in Franz cells.")
    store = Store(db_path())
    ingest_path(d, store, Config())
    M.remember(store, "Rejected: Ussing chamber for timolol, edge damage", "rejected", cwd=str(d))
    M.remember(store, "cursor session; goal: upload the model figures to google drive and retry the "
                      "timolol export; edited 24 files; ran: head, ps, grep", "session", cwd=str(d))
    q = "timolol corneal permeability Franz model"            # the session log shares 2 of these 5
    mix = context(q, store, Config(), mode="mix", max_tokens=2000, cwd=str(d))
    assert [m["kind"] for m in mix["memories"]] == ["rejected"]
    everything = context(q, store, Config(), mode="memory", max_tokens=2000, cwd=str(d))
    assert {m["kind"] for m in everything["memories"]} == {"rejected", "session"}


def test_naming_a_project_finds_its_code_even_when_the_code_never_says_the_name(tmp_path):
    root = tmp_path / "repos"
    _repo(root, "karpathy__nanoGPT", "class CausalSelfAttention(nn.Module):\n    # masked attention block\n")
    big = root / "transformers"
    big.mkdir(parents=True)
    (big / ".git").mkdir()
    for i in range(30):
        (big / f"m{i}.py").write_text(f"class Attention{i}(nn.Module):\n    # causal attention block {i}\n")
    store = Store(db_path())
    ingest_path(root, store, Config())
    hits = retrieve("where is the causal attention block defined in nanoGPT", store, Config(), top_k=5).chunks
    assert hits[0]["project_name"] == "karpathy__nanoGPT"
    plain = retrieve("where is the causal attention block defined", store, Config(), top_k=5).chunks
    assert all(h["project_name"] == "transformers" for h in plain)    # no name, no boost


def test_plain_folder_names_are_not_project_names(tmp_path):
    from cartograph.retrieve import _short_name
    assert _short_name("References/frontier_pc/pallets__flask") == "flask"
    assert _short_name("References/frontier_pc/karpathy__nanoGPT") == "nanogpt"
    assert _short_name("Tools/AutoGPT") == "autogpt"
    for plain in ("Notes", "outputs", "Projects/data", "Code", "Work", "langchain"):
        assert _short_name(plain) is None


def test_question_words_do_not_drive_keyword_ranking(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    (d / ".git").mkdir()
    (d / "a.md").write_text("How do I do this with that? What is it, and where is it from, and how?")
    (d / "b.md").write_text("Position sizing scales each trade to a volatility target.")
    store = Store(db_path())
    ingest_path(d, store, Config())
    hits = retrieve("how do I size positions with volatility targeting", store, Config()).chunks
    assert hits and hits[0]["file_path"].endswith("b.md")
    assert all(not h["file_path"].endswith("a.md") for h in hits)
