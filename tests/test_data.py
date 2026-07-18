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
