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


def test_sample_measurement_matches_bootstrap_S_diag():
    from src.talent3 import sample_measurement, FLOOR_SD_PER_PA
    rng = np.random.default_rng(0)
    v = np.array([1.2, 0.1, 0.69, 0.0, 2.0, 0.0]); d = np.ones_like(v)
    z, s00 = sample_measurement(v, d, B=500, rng=rng)
    assert abs(z - v.sum() / d.sum()) < 1e-12
    # equals bootstrap_S[0,0] under the same seed
    from src.talent2 import bootstrap_S
    nan = np.full_like(v, np.nan)
    S = bootstrap_S(v, d, nan, nan, B=500, rng=np.random.default_rng(0))
    assert abs(s00 - S[0, 0]) < 1e-12


def test_sample_measurement_floor_binds_on_degenerate_sample():
    from src.talent3 import sample_measurement, FLOOR_SD_PER_PA
    rng = np.random.default_rng(1)
    v = np.array([0.0, 0.0]); d = np.array([1.0, 1.0])   # zero variation
    _, s00 = sample_measurement(v, d, B=200, rng=rng)
    assert abs(s00 - FLOOR_SD_PER_PA ** 2 / 2) < 1e-12   # floored, not zero


def test_season_mu_causal_uses_only_first_k():
    from src.talent3 import season_mu_causal
    # Two players, 2024. With k=1, only each player's earliest PA counts.
    f = build_pa_frame(_pitches()).filter(pl.col("season") == 2024)
    # player1 earliest (04-01) value .1 ; player2 earliest (04-01) value .8
    mu_k1 = season_mu_causal(f, season=2024, k=1)
    assert abs(mu_k1 - (0.1 + 0.8) / 2) < 1e-12
    # full-season (k huge) = pooled league rate over all PAs
    mu_full = season_mu_causal(f, season=2024, k=10_000)
    all_v = f["value"].to_numpy(); all_d = f["denom"].to_numpy()
    assert abs(mu_full - all_v.sum() / all_d.sum()) < 1e-12


from src.talent3 import fit_hypers_eb


def test_fit_hypers_recovers_known_variances():
    rng = np.random.default_rng(7)
    sig_eta, sig_u = 0.030, 0.015
    # 400 players, 3 seasons each, known measurement noise S≈0.02² per obs
    ys, Ss, pid = [], [], []
    for i in range(400):
        eta = rng.normal(0, sig_eta)
        for s in range(3):
            u = rng.normal(0, sig_u)
            S = 0.02 ** 2
            ys.append(eta + u + rng.normal(0, np.sqrt(S)))
            Ss.append(S); pid.append(i)
    est_eta2, est_u2 = fit_hypers_eb(np.array(ys), np.array(Ss), np.array(pid))
    assert abs(np.sqrt(est_eta2) - sig_eta) < 0.006   # within ~1 sampling SE
    assert abs(np.sqrt(est_u2) - sig_u) < 0.006
