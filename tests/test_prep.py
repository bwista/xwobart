import polars as pl
import pytest

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


from src.prep import stratified_subsample


def _imbalanced(n=10_000):
    # ~66% out, 21% single, 8% double, 0.7% triple, 4.3% HR — roughly real proportions
    counts = {0: 6600, 1: 2100, 2: 800, 3: 70, 4: 430}
    rows = [c for c, k in counts.items() for _ in range(k)]
    return pl.DataFrame({"outcome_class": rows, "payload": list(range(len(rows)))})


def test_subsample_preserves_proportions_and_size():
    df = _imbalanced()
    sub = stratified_subsample(df, 1000, seed=42)
    assert sub.height == 1000
    got = dict(sub.group_by("outcome_class").len().iter_rows())
    assert got[3] in (6, 7, 8)            # triples survive, proportionally (~7)
    assert abs(got[0] - 660) <= 2         # never rebalanced


def test_subsample_reproducible_and_noop_when_large():
    df = _imbalanced()
    a = stratified_subsample(df, 500, seed=1)["payload"].to_list()
    b = stratified_subsample(df, 500, seed=1)["payload"].to_list()
    assert a == b
    assert stratified_subsample(df, 10_000_000, seed=1).height == df.height
    assert stratified_subsample(df, None, seed=1).height == df.height


from src.prep import HOME_PLATE_X, HOME_PLATE_Y, add_spray, spray_impute_table


def _spray_df():
    """Four BBE with hand-computed geometry, plus one with hc missing.

    hc_x < 125.42 is LEFT field, hc_x > 125.42 is RIGHT field; hc_y DECREASES going
    out toward the outfield, so 198.27 - hc_y > 0 for any ball in play.
    """
    return pl.DataFrame({
        "batter":       [1, 2, 3, 4, 5],
        "game_year":    [2024] * 5,
        "stand":        ["R", "L", "R", "L", "R"],
        # 45 deg to the LEFT, 45 deg to the RIGHT, dead center, 45 left, missing
        "hc_x":         [75.42, 175.42, 125.42, 75.42, None],
        "hc_y":         [148.27, 148.27, 148.27, 148.27, None],
        "launch_speed": [100.0, 100.0, 100.0, 100.0, 100.0],
        "launch_angle": [25.0, 25.0, 25.0, 25.0, 25.0],
    })


def test_add_spray_sign_convention_is_pull_relative():
    cell, hand = spray_impute_table(_spray_df())
    out = add_spray(_spray_df(), cell, hand).sort("batter")
    # raw direction: negative = left field, positive = right field
    assert abs(out["phi_raw"][0] - (-45.0)) < 1e-9      # RHB pulled to left
    assert abs(out["phi_raw"][1] - (+45.0)) < 1e-9      # LHB pulled to right
    assert abs(out["phi_raw"][2] - 0.0) < 1e-9          # dead center
    assert abs(out["phi_raw"][3] - (-45.0)) < 1e-9      # LHB went OPPOSITE field
    # pull-relative: POSITIVE means pulled, for BOTH hands
    assert abs(out["spray_pull"][0] - (+45.0)) < 1e-9   # RHB to left  = pulled
    assert abs(out["spray_pull"][1] - (+45.0)) < 1e-9   # LHB to right = pulled
    assert abs(out["spray_pull"][2] - 0.0) < 1e-9
    assert abs(out["spray_pull"][3] - (-45.0)) < 1e-9   # LHB to left  = opposite
    assert out["stand_R"].to_list() == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_add_spray_mirror_is_not_identity_or_global_flip():
    """Guards the two mirror bugs a single sign test cannot distinguish."""
    out = add_spray(_spray_df(), *spray_impute_table(_spray_df())).sort("batter")
    phi, sp = out["phi_raw"].to_numpy()[:4], out["spray_pull"].to_numpy()[:4]
    assert not np.allclose(sp, phi)          # not "forgot to mirror"
    assert not np.allclose(sp, -phi)         # not "mirrored both hands"


def test_add_spray_imputes_missing_hc_and_flags_without_dropping():
    df = _spray_df()
    cell, hand = spray_impute_table(df)
    out = add_spray(df, cell, hand).sort("batter")
    assert out.height == df.height                      # NEVER drop: ELPD anchor n
    assert out["spray_pull"].null_count() == 0
    assert out["hc_imputed"].to_list() == [False, False, False, False, True]
    # RHB with no hc falls back to the RHB median (cells need >= 25 rows, so the
    # per-hand fallback is what fires here): median of the two RHB spray values
    assert abs(out["spray_pull"][4] - float(np.median([45.0, 0.0]))) < 1e-9


def test_spray_impute_table_uses_cells_when_dense_enough():
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame({
        "batter": list(range(n)), "game_year": [2024] * n, "stand": ["R"] * n,
        # all in one (ev_bin, la_bin) cell, all pulled to left field
        "hc_x": (125.42 - rng.uniform(20, 60, n)).tolist(),
        "hc_y": [148.27] * n,
        "launch_speed": [101.0] * n, "launch_angle": [26.0] * n,
    })
    cell, hand = spray_impute_table(df)
    assert cell.height == 1 and cell["spray_cell"][0] > 0     # pulled, positive
    miss = df.head(1).with_columns(hc_x=pl.lit(None, dtype=pl.Float64),
                                   hc_y=pl.lit(None, dtype=pl.Float64))
    out = add_spray(miss, cell, hand)
    assert out["hc_imputed"][0] and abs(out["spray_pull"][0] - cell["spray_cell"][0]) < 1e-9


def test_add_spray_rejects_null_stand():
    bad = _spray_df().with_columns(stand=pl.lit(None, dtype=pl.Utf8))
    with pytest.raises(AssertionError, match="stand"):
        add_spray(bad, *spray_impute_table(_spray_df()))
