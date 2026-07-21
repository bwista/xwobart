import numpy as np
import polars as pl
from src.talent3 import build_pa_frame


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
