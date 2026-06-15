"""Vision pipeline: privacy (denylist + redaction), the novelty/similarity cache, classification,
ingestion into the graph, the watch loop, and live context in the brief — all with fakes (no screen)."""
from cartograph.config import Config
from cartograph.storage import Store
from cartograph.vision.capture import FakeCapturer, Frame
from cartograph.vision.dedup import NoveltyGate
from cartograph.vision.pipeline import VisionConfig, process_frame, recent_activity
from cartograph.vision.redact import is_sensitive_window, redact
from cartograph.vision.watch import watch

CODE = ("def train_model(x):\n    import torch\n    model = torch.nn.Linear(10, 2)\n"
        "    return model  # neural network training loop with transformer layers\n") * 3


def _frame(title="VS Code — train.py"):
    return Frame(image=None, app_title=title, ts=0.0)            # image=None -> text-only path


def _gate():
    return NoveltyGate()


def test_redaction_strips_secrets():
    txt = "email me at a@b.com, key sk-ABCDEFGHIJKL1234, pwd password: hunter2"
    clean, n = redact(txt)
    assert "a@b.com" not in clean and "sk-ABCDEFGHIJKL1234" not in clean and "hunter2" not in clean
    assert n >= 3


def test_sensitive_window_denylisted():
    assert is_sensitive_window("Chase Bank — Online Banking", None)
    assert is_sensitive_window("1Password", None)
    assert not is_sensitive_window("VS Code — main.py", None)


def test_sensitive_window_never_processed(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    rec = process_frame(_frame("My Bank — Login"), store, Config(), VisionConfig(), _gate(),
                        ocr=lambda _img: CODE, apply=True)
    assert rec["action"] == "skip" and rec["reason"] == "sensitive_window"
    assert store.project_id_by_name("screen-activity") is None     # nothing stored


def test_novelty_cache_skips_duplicate_text(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    gate, cfg, vcfg = _gate(), Config(), VisionConfig()
    r1 = process_frame(_frame(), store, cfg, vcfg, gate, ocr=lambda _i: CODE, apply=True)
    r2 = process_frame(_frame(), store, cfg, vcfg, gate, ocr=lambda _i: CODE, apply=True)   # identical
    assert r1["action"] == "store"
    assert r2["action"] == "skip" and r2["reason"] == "duplicate_text"


def test_too_little_text_skipped(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    rec = process_frame(_frame(), store, Config(), VisionConfig(), _gate(),
                        ocr=lambda _i: "hi", apply=True)
    assert rec["action"] == "skip" and rec["reason"] == "too_little_text"


def test_store_classifies_and_ingests(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    rec = process_frame(_frame(), store, Config(), VisionConfig(), _gate(),
                        ocr=lambda _i: CODE, apply=True)
    assert rec["action"] == "store"
    assert rec["intent"] in {"debug", "code", "review", "generate", "explain", "general"}
    act = recent_activity(store, limit=1)                          # parsed back out of the graph
    assert act and act[0]["app"].startswith("VS Code")


def test_dry_run_stores_nothing(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    rec = process_frame(_frame(), store, Config(), VisionConfig(), _gate(),
                        ocr=lambda _i: CODE, apply=False)
    assert rec["action"] == "preview"
    assert store.project_id_by_name("screen-activity") is None     # dry run: nothing persisted


def test_watch_loop_runs_finite(tmp_path):
    store = Store(tmp_path / "g.sqlite")
    cap = FakeCapturer([(None, "Editor — a.py"), (None, "Editor — a.py")])
    summary = watch(store, Config(), VisionConfig(), cap, ocr=lambda _i: CODE,
                    iterations=2, apply=True, sleep=lambda *_: None)
    assert summary["ticks"] == 2
