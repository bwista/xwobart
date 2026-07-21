import numpy as np
import polars as pl
from src.talent3 import build_pa_frame, cutpoint_split


def _pitches():
    return pl.DataFrame({
        "batter":     [1, 1, 1, 2, 2, 1],
        "game_year":  [2024, 2024, 2024, 2024, 2024, 2024],
        "game_date":  ["2024-04-02", "2024-04-01", "2024-05-01",
                       "2024-04-01", "2024-04-02", "2024-04-03"],
        "type":       ["X", "X", "B", "X", "S", "S"],
        "estimated_woba_using_speedangle": [1.2, 0.1, None, 0.8, None, None],
        "woba_value": [1.25, 0.0, 0.69, 0.9, 0.0, 0.0],
        "woba_denom": [1, 1, 1, 1, 1, None],   # last row: non-PA, dropped
    })


def test_build_pa_frame_keeps_date_and_value_logic():
    f = build_pa_frame(_pitches())
    assert set(f.columns) == {"batter", "season", "game_date", "value", "denom"}
    assert f.height == 5                              # non-PA row dropped
    p1 = f.filter(pl.col("batter") == 1).sort("value")
    assert p1["value"].to_list() == [0.1, 0.69, 1.2]  # BBE→est_woba, walk→woba_value
    assert f["game_date"].dtype == pl.Date            # parsed for ordering


def test_cutpoint_split_orders_and_blends():
    f = (build_pa_frame(_pitches()).filter(pl.col("batter") == 2))  # 2 PAs: .8, .0
    # k=1 → observed = earliest PA (2024-04-01 value .8), remaining = .0
    cp = cutpoint_split(f, k=1, min_remaining=1)
    assert cp is not None
    assert abs(cp["r_obs"] - 0.8) < 1e-12 and cp["D_obs"] == 1
    assert abs(cp["r_rest"] - 0.0) < 1e-12 and cp["D_rest"] == 1
    assert abs(cp["w"] - 0.5) < 1e-12
    # blend identity holds against the whole-season rate
    from src.forecast import final_line_blend
    r_final, _ = final_line_blend(cp["r_obs"], cp["D_obs"], cp["r_rest"], cp["D_rest"])
    assert abs(r_final - (0.8 + 0.0) / 2) < 1e-12


def test_cutpoint_split_ineligible_when_no_runway():
    f = build_pa_frame(_pitches()).filter(pl.col("batter") == 2)
    assert cutpoint_split(f, k=1, min_remaining=5) is None   # only 1 remaining PA
    assert cutpoint_split(f, k=2, min_remaining=1) is None   # nothing remaining
