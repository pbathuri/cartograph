from cartograph.cli import _clean_fields, _clean_roots


def test_clean_roots_strips_escape_codes_and_missing(tmp_path):
    real = tmp_path / "Work"
    real.mkdir()
    good, bad = _clean_roots([f"\x1b[A\x1b[B{real}", str(tmp_path / "nope"), str(real)])
    assert good == [str(real)]  # escape junk stripped, duplicate collapsed
    assert len(bad) == 1


def test_clean_fields_maps_free_text():
    good, bad = _clean_fields(["Finance", "quant", "ML experiment", "astrology"])
    assert good == ["finance_accounting", "quant_research", "ml_experiment"]
    assert bad == ["astrology"]
