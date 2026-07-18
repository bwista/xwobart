from scripts.run_v0 import update_results_md


def test_results_section_idempotent(tmp_path):
    update_results_md(tmp_path, "A", ["runtime: 10s"])
    update_results_md(tmp_path, "B", ["runtime: 100s"])
    update_results_md(tmp_path, "A", ["runtime: 12s"])   # replaces, not appends
    text = (tmp_path / "RESULTS.md").read_text()
    assert text.count("<!-- stage_A -->") == 1
    assert "runtime: 12s" in text and "runtime: 10s" not in text
    assert "runtime: 100s" in text
