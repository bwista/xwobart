import pytest

from src.data import _retry, coverage_gaps


def test_coverage_complete():
    assert coverage_gaps(2024, "2024-03-20", "2024-09-30", ("2024-03-20", "2024-09-30")) == []


def test_coverage_missing_tail_and_head():
    gaps = coverage_gaps(2022, "2022-04-08", "2022-09-30", ("2022-04-07", "2022-10-05"))
    assert len(gaps) == 2
    assert any("ends 2022-09-30" in g for g in gaps)


def test_retry_succeeds_after_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert _retry(flaky, tries=3, wait_s=0) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhaustion():
    def always():
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        _retry(always, tries=2, wait_s=0)


import polars as pl

from src.data import merge_sprint_speed


def test_sprint_merge_and_median_imputation():
    bbe = pl.DataFrame({"batter": [10, 20, 30], "game_year": [2024, 2024, 2024]})
    sprint = pl.DataFrame({
        "player_id": [10, 40, 50], "season": [2024, 2024, 2024],
        "sprint_speed": [30.0, 26.0, 28.0],
    })
    out, rate = merge_sprint_speed(bbe, sprint)
    row = {b: (s, i) for b, s, i in zip(out["batter"], out["sprint_speed"], out["imputed_speed"])}
    assert row[10] == (30.0, False)          # matched
    assert row[20] == (28.0, True)           # imputed to league median (median of 30,26,28)
    assert row[30] == (28.0, True)
    assert abs(rate["imputation_rate"][0] - 2 / 3) < 1e-9


def test_sprint_merge_is_per_season():
    bbe = pl.DataFrame({"batter": [10, 10], "game_year": [2023, 2024]})
    sprint = pl.DataFrame({
        "player_id": [10, 10], "season": [2023, 2024],
        "sprint_speed": [29.0, 27.0],
    })
    out, _ = merge_sprint_speed(bbe, sprint)
    assert out.sort("game_year")["sprint_speed"].to_list() == [29.0, 27.0]


from src.data import KEEP_COLUMNS, cache_fingerprint

PRE_REBUILD_COLUMNS = [
    "game_pk", "game_date", "game_year", "batter", "events", "description",
    "des", "type", "bb_type", "launch_speed", "launch_angle",
    "launch_speed_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
]


def test_keep_columns_adds_spray_inputs_without_removing_anything():
    assert KEEP_COLUMNS[:len(PRE_REBUILD_COLUMNS)] == PRE_REBUILD_COLUMNS
    assert KEEP_COLUMNS[len(PRE_REBUILD_COLUMNS):] == ["hc_x", "hc_y", "stand"]


def _fp_frame(tmp_path, extra=None, vals=(1, 2, 3)):
    d = {"a": list(vals), "b": ["x", "y", "z"]}
    if extra:
        d |= extra
    # The filename MUST discriminate on content: two frames differing only in `vals`
    # would otherwise collide and silently overwrite each other, making the
    # changed-value assertion below compare a file to itself and always fail.
    p = tmp_path / f"f{len(d)}_{'-'.join(map(str, vals))}.parquet"
    pl.DataFrame(d).write_parquet(p)
    return p


def test_fingerprint_ignores_added_columns(tmp_path):
    p1 = _fp_frame(tmp_path)
    p2 = _fp_frame(tmp_path, extra={"c": [9.0, 9.0, 9.0]})
    assert cache_fingerprint(p1, ["a", "b"]) == cache_fingerprint(p2, ["a", "b"])


def test_fingerprint_ignores_row_order(tmp_path):
    p1 = _fp_frame(tmp_path, vals=(1, 2, 3))
    p2 = tmp_path / "rev.parquet"
    pl.read_parquet(p1).reverse().write_parquet(p2)
    assert cache_fingerprint(p1, ["a", "b"]) == cache_fingerprint(p2, ["a", "b"])


def test_fingerprint_detects_a_changed_value_and_a_changed_row_count(tmp_path):
    p1 = _fp_frame(tmp_path, vals=(1, 2, 3))
    p2 = _fp_frame(tmp_path, vals=(1, 2, 4))
    assert cache_fingerprint(p1, ["a", "b"])["digest"] != cache_fingerprint(p2, ["a", "b"])["digest"]
    p3 = tmp_path / "short.parquet"
    pl.read_parquet(p1).head(2).write_parquet(p3)
    assert cache_fingerprint(p3, ["a", "b"])["n_rows"] == 2
