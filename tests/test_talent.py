import numpy as np
import polars as pl

from src.talent import build_pa_values, per_player_raw


def _pitches():
    # 2 players, 2024. type X uses est_woba; walk/K use woba_value; non-PA rows dropped.
    return pl.DataFrame({
        "batter":       [1, 1, 1, 2, 2, 1],
        "game_year":    [2024, 2024, 2024, 2024, 2024, 2024],
        "type":         ["X", "X", "B", "X", "S", "S"],
        "estimated_woba_using_speedangle": [1.2, 0.1, None, 0.8, None, None],
        "woba_value":   [1.25, 0.0, 0.69, 0.9, 0.0, 0.0],
        "woba_denom":   [1, 1, 1, 1, 1, None],   # last row is a non-PA pitch -> dropped
    })


def test_build_pa_values_picks_est_for_bbe_and_woba_else():
    pav = build_pa_values(_pitches()).sort("batter", "value")
    # player 1: three PAs -> values 1.2 (X est), 0.1 (X est), 0.69 (walk woba_value)
    v1 = pav.filter(pl.col("batter") == 1)["value"].sort().to_list()
    assert v1 == [0.1, 0.69, 1.2]
    # non-PA row (woba_denom null) dropped: player 1 has 3 PAs, player 2 has 2
    # (batted ball + strikeout, type S woba_denom 1) -> 5 PAs across both players
    assert pav.height == 5
    assert set(pav.columns) == {"batter", "season", "value", "denom"}


def test_per_player_raw_xwoba_and_se():
    pav = build_pa_values(_pitches())
    raw = per_player_raw(pav).sort("batter")
    r1 = raw.filter(pl.col("batter") == 1).row(0, named=True)
    assert r1["PA"] == 3
    assert abs(r1["xwoba_raw"] - (1.2 + 0.1 + 0.69) / 3) < 1e-9
    # se = sample sd of the three values / sqrt(3)
    vals = np.array([1.2, 0.1, 0.69])
    assert abs(r1["se"] - vals.std(ddof=1) / np.sqrt(3)) < 1e-9
    assert r1["se2"] > 0
