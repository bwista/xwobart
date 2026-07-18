import polars as pl

from src.prep import add_outcome_class, class_distribution, filter_bbe


def _pitch_df():
    # 6 rows: 1 non-X, 1 bunt-X, 1 missing-LA X, 3 clean X
    return pl.DataFrame({
        "game_year":     [2024, 2024, 2024, 2024, 2024, 2023],
        "type":          ["S", "X", "X", "X", "X", "X"],
        "des":           ["swinging strike", "Jose Altuve sacrifice bunts.", None, "flies out", "homers", "singles"],
        "launch_speed":  [None, 60.0, 95.0, 88.0, 105.0, 91.0],
        "launch_angle":  [None, 10.0, None, 25.0, 30.0, 5.0],
        "events":        [None, "sac_bunt", "single", "field_out", "home_run", "single"],
    })


def test_filter_bbe_excludes_non_x_bunts_and_missing():
    bbe, report = filter_bbe(_pitch_df())
    assert bbe.height == 3
    assert set(bbe["events"].to_list()) == {"field_out", "home_run", "single"}
    r24 = report.filter(pl.col("game_year") == 2024).row(0, named=True)
    assert r24["n_bbe_raw"] == 4 and r24["n_bunt"] == 1 and r24["n_missing_ls_la"] == 1


def test_outcome_class_mapping():
    df = pl.DataFrame({"events": [
        "single", "double", "triple", "home_run",
        "field_out", "field_error", "fielders_choice", "sac_fly",
        "grounded_into_double_play",
    ]})
    out = add_outcome_class(df)
    assert out["outcome_class"].to_list() == [1, 2, 3, 4, 0, 0, 0, 0, 0]


def test_class_distribution_sums_to_100():
    df = add_outcome_class(pl.DataFrame({"events": ["single"] * 3 + ["field_out"] * 7}))
    dist = class_distribution(df)
    assert abs(dist["pct"].sum() - 100.0) < 1e-6
    assert dist.filter(pl.col("outcome_class") == 1)["len"].item() == 3
