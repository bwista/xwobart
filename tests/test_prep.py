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


import numpy as np

from src.prep import FEATURES, build_features, build_non_bbe_pa


def test_build_features_exact_three_columns():
    df = pl.DataFrame({
        "launch_speed": [90.0, 100.0], "launch_angle": [10.0, 25.0],
        "sprint_speed": [27.0, 29.5], "outcome_class": [0, 4],
        "extra": ["a", "b"],
    })
    X, y = build_features(df)
    assert X.shape == (2, 3) and X.dtype == np.float64
    assert FEATURES == ["launch_speed", "launch_angle", "sprint_speed"]
    assert y.tolist() == [0, 4] and y.dtype == np.int64
    assert X[1, 2] == 29.5


def test_non_bbe_pa_table():
    df = pl.DataFrame({
        "batter":     [1, 1, 2, 3],
        "game_year":  [2024, 2024, 2024, 2024],
        "type":       ["S", "X", "B", "S"],
        "woba_value": [0.0, 0.9, 0.7, None],
        "woba_denom": [1, 1, 1, None],
    })
    out = build_non_bbe_pa(df)
    # row 0: K (kept). row 1: type X (excluded). row 2: walk (kept). row 3: null denom (excluded).
    assert out.height == 2
    assert out.columns == ["batter", "season", "woba_value", "woba_denom"]
    assert out["woba_value"].to_list() == [0.0, 0.7]
