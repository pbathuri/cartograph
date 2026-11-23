"""Change detection hashes the whole file (an edit past the first MB used to go unnoticed), and an
unchanged file — same size and mtime — is skipped without being read."""
import os

from cartograph import ingest as ing
from cartograph.config import Config, db_path
from cartograph.storage import Store


def _proj(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "pyproject.toml").write_text("[project]\nname='p'\n")
    return d


def _chunks_of(store, path):
    with store.cursor() as c:
        return [r[0] for r in c.execute(
            "SELECT ch.chunk_text FROM chunks ch JOIN files f ON f.id=ch.file_id WHERE f.path=?", (str(path),))]


def test_edit_after_first_megabyte_is_detected(tmp_path):
    d = _proj(tmp_path)
    big = d / "notes.md"
    head = ("alpha line\n" * 100_000)[:1_100_000]
    big.write_text(head + "\nTAILMARKER one\n")
    cfg, store = Config(max_file_mb=2.0), Store(db_path())
    ing.ingest_path(d, store, cfg)
    assert any("TAILMARKER one" in t for t in _chunks_of(store, big))
    st = big.stat()
    big.write_text(head + "\nTAILMARKER two\n")
    os.utime(big, (st.st_atime, st.st_mtime + 10))
    ing.ingest_path(d, store, cfg)
    texts = _chunks_of(store, big)
    assert any("TAILMARKER two" in t for t in texts) and not any("TAILMARKER one" in t for t in texts)


def test_unchanged_files_are_not_read(tmp_path, monkeypatch):
    d = _proj(tmp_path)
    (d / "a.md").write_text("gradient descent notes\n")
    cfg, store = Config(), Store(db_path())
    ing.ingest_path(d, store, cfg)
    reads = []
    real = ing._hash
    monkeypatch.setattr(ing, "_hash", lambda p: reads.append(p) or real(p))
    st = ing.ingest_path(d, store, cfg)
    assert reads == [] and st.files_indexed == 0


def test_legacy_rows_upgrade_without_rechunking(tmp_path):
    d = _proj(tmp_path)
    f = d / "a.md"
    f.write_text("walk forward backtest\n")
    cfg, store = Config(), Store(db_path())
    ing.ingest_path(d, store, cfg)
    with store.cursor() as c:
        c.execute("UPDATE files SET size=NULL, mtime=NULL")
        before = [r[0] for r in c.execute("SELECT id FROM chunks")]
    st = ing.ingest_path(d, store, cfg)
    with store.cursor() as c:
        after = [r[0] for r in c.execute("SELECT id FROM chunks")]
        filled = c.execute("SELECT count(*) FROM files WHERE size IS NOT NULL AND mtime IS NOT NULL").fetchone()[0]
    assert before == after and st.files_indexed == 0 and filled >= 1


def test_migration_adds_columns_to_an_old_database(tmp_path):
    import sqlite3
    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT, ext TEXT, hash TEXT,"
                " UNIQUE(project_id, path))")
    con.commit()
    con.close()
    Store(p)
    cols = {r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(files)")}
    assert {"size", "mtime"} <= cols
