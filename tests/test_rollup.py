import numpy as np
import polars as pl

from src.rollup import expected_values, linear_weights


def test_linear_weights_empirical_means():
    bbe = pl.DataFrame({
        "outcome_class": [0, 0, 0, 1, 1, 2, 4],
        "woba_value":    [0.0, 0.0, 0.9, 0.9, 0.9, 1.25, 2.0],  # one ROE in the out class
    })
    w, warnings = linear_weights(bbe)
    assert w.shape == (5,)
    assert abs(w[0] - 0.3) < 1e-9          # empirical mean, NOT forced 0 (spec §8.1)
    assert w[1] == 0.9 and w[2] == 1.25 and w[3] == 0.0 and w[4] == 2.0
    assert any("w[0]" in msg for msg in warnings)   # 0.3 is outside the sanity range
    assert any("w[3]" in msg for msg in warnings)   # empty triple class -> 0.0 flagged


def test_expected_values_dot():
    w = np.array([0.0, 1.0, 0.0, 0.0, 2.0])
    p = np.array([  # (S=2, K=5, n=2)
        [[0.5, 0.0], [0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.5], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    ev = expected_values(p, w)
    assert ev.shape == (2, 2)
    assert np.allclose(ev, [[0.5, 2.0], [0.0, 0.5]])
    assert ev.dtype == np.float32


from src.rollup import player_rollup


def test_player_rollup_hand_checked():
    # Player 7, season 2024: two BBE + one strikeout (value 0, denom 1).
    # draw 0: (0.5 + 1.0 + 0) / 3 = 0.5 ; draw 1: (0.3 + 0.7 + 0) / 3 = 1/3
    ev = np.array([[0.5, 1.0], [0.3, 0.7]], dtype=np.float32)      # (S=2, n=2)
    bbe_keys = pl.DataFrame({"batter": [7, 7], "season": [2024, 2024], "woba_denom": [1.0, 1.0]})
    non_bbe = pl.DataFrame({"batter": [7], "season": [2024], "woba_value": [0.0], "woba_denom": [1.0]})
    out = player_rollup(ev, bbe_keys, non_bbe)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["PA"] == 3
    assert abs(row["xwoba_mean"] - (0.5 + 1 / 3) / 2) < 1e-6
    assert row["xwoba_q05"] < row["xwoba_mean"] < row["xwoba_q95"]


def test_player_rollup_includes_non_bbe_only_players():
    # Player 9 only ever walks — must still appear (all-walk seasons exist).
    ev = np.array([[0.5]], dtype=np.float32)
    bbe_keys = pl.DataFrame({"batter": [7], "season": [2024], "woba_denom": [1.0]})
    non_bbe = pl.DataFrame({"batter": [9], "season": [2024], "woba_value": [0.7], "woba_denom": [1.0]})
    out = player_rollup(ev, bbe_keys, non_bbe).sort("batter")
    assert out["batter"].to_list() == [7, 9]
    assert abs(out.filter(pl.col("batter") == 9)["xwoba_mean"].item() - 0.7) < 1e-9


def test_player_rollup_row_alignment_after_join():
    # Two players interleaved — verifies ev rows stay aligned with bbe_keys order.
    ev = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    bbe_keys = pl.DataFrame({
        "batter": [1, 2, 1, 2], "season": [2024] * 4, "woba_denom": [1.0] * 4,
    })
    non_bbe = pl.DataFrame({"batter": [], "season": [], "woba_value": [], "woba_denom": []},
                           schema={"batter": pl.Int64, "season": pl.Int64,
                                   "woba_value": pl.Float64, "woba_denom": pl.Float64})
    out = player_rollup(ev, bbe_keys, non_bbe).sort("batter")
    assert out.filter(pl.col("batter") == 1)["xwoba_mean"].item() == 1.0
    assert out.filter(pl.col("batter") == 2)["xwoba_mean"].item() == 0.0
