import numpy as np
import polars as pl

from src.talent2 import build_pa_measurements


def _pitches():
    # 3 players' PAs, 2024. Rows: BBE tracked, BBE tracked (barrel), walk,
    # BBE untracked (null EV/LSA -> excluded from peripherals but keeps its value),
    # strikeout, non-PA pitch (dropped).
    return pl.DataFrame({
        "batter":       [1,    1,    1,    2,    2,    1],
        "game_year":    [2024, 2024, 2024, 2024, 2024, 2024],
        "type":         ["X",  "X",  "B",  "X",  "S",  "S"],
        "launch_speed": [101.3, 88.0, None, None, None, None],
        "launch_speed_angle": [6.0, 3.0, None, None, None, None],
        "estimated_woba_using_speedangle": [1.2, 0.1, None, 0.8, None, None],
        "woba_value":   [1.25, 0.0, 0.69, 0.9, 0.0, 0.0],
        "woba_denom":   [1,    1,    1,    1,    1,    None],
    })


def test_build_pa_measurements_values_match_phase1_logic():
    pam = build_pa_measurements(_pitches())
    # non-PA row dropped: 5 rows; value logic identical to talent.build_pa_values
    assert pam.height == 5
    v1 = pam.filter(pl.col("batter") == 1)["value"].sort().to_list()
    assert v1 == [0.1, 0.69, 1.2]
    assert set(pam.columns) == {"batter", "season", "value", "denom", "ev", "barrel"}


def test_build_pa_measurements_ev_barrel_only_on_tracked_bbe():
    pam = build_pa_measurements(_pitches()).sort("batter", "value")
    p1 = pam.filter(pl.col("batter") == 1).sort("value")
    # walk row: ev/barrel null; tracked BBE rows carry ev and barrel = (lsa == 6)
    assert p1.filter(pl.col("value") == 0.69)["ev"][0] is None
    assert p1.filter(pl.col("value") == 1.2)["ev"][0] == 101.3
    assert p1.filter(pl.col("value") == 1.2)["barrel"][0] == 1.0
    assert p1.filter(pl.col("value") == 0.1)["barrel"][0] == 0.0
    # player 2's BBE has null launch_speed/lsa -> untracked: ev AND barrel null,
    # but the PA still contributes its xwOBA value
    p2x = pam.filter((pl.col("batter") == 2) & (pl.col("value") == 0.8))
    assert p2x["ev"][0] is None and p2x["barrel"][0] is None
