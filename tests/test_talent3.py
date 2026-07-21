import numpy as np
import polars as pl
import pytest

from src.forecast import final_line_blend
from src.talent2 import bootstrap_S
from src.talent3 import (
    FLOOR_SD_PER_PA,
    build_pa_frame,
    cutpoint_posterior,
    cutpoint_split,
    fit_hypers_eb,
    sample_measurement,
    season_mu_causal,
)


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
    r_final, _ = final_line_blend(cp["r_obs"], cp["D_obs"], cp["r_rest"], cp["D_rest"])
    assert abs(r_final - (0.8 + 0.0) / 2) < 1e-12


def test_cutpoint_split_ineligible_when_no_runway():
    f = build_pa_frame(_pitches()).filter(pl.col("batter") == 2)
    assert cutpoint_split(f, k=1, min_remaining=5) is None   # only 1 remaining PA
    assert cutpoint_split(f, k=2, min_remaining=1) is None   # nothing remaining


def test_cutpoint_split_rejects_negative_k():
    f = build_pa_frame(_pitches()).filter(pl.col("batter") == 2)
    with pytest.raises(AssertionError):
        cutpoint_split(f, k=-1, min_remaining=1)


def test_cutpoint_split_none_when_remaining_denom_zero():
    # remaining PA has denom 0 -> D_rest == 0; even with min_remaining=0 there is no rest
    frame = pl.DataFrame({
        "batter": [7, 7], "season": [2024, 2024],
        "game_date": pl.Series(["2024-04-01", "2024-04-02"]).str.to_date(),
        "value": [0.5, 0.0], "denom": [1.0, 0.0],
    })
    assert cutpoint_split(frame, k=1, min_remaining=0) is None


def test_sample_measurement_matches_bootstrap_S_diag():
    rng = np.random.default_rng(0)
    v = np.array([1.2, 0.1, 0.69, 0.0, 2.0, 0.0]); d = np.ones_like(v)
    z, s00 = sample_measurement(v, d, B=500, rng=rng)
    assert abs(z - v.sum() / d.sum()) < 1e-12
    # equals bootstrap_S[0,0] under the same seed
    nan = np.full_like(v, np.nan)
    S = bootstrap_S(v, d, nan, nan, B=500, rng=np.random.default_rng(0))
    assert abs(s00 - S[0, 0]) < 1e-12


def test_sample_measurement_floor_binds_on_degenerate_sample():
    rng = np.random.default_rng(1)
    v = np.array([0.0, 0.0]); d = np.array([1.0, 1.0])   # zero variation
    _, s00 = sample_measurement(v, d, B=200, rng=rng)
    assert abs(s00 - FLOOR_SD_PER_PA ** 2 / 2) < 1e-12   # floored, not zero


def test_season_mu_causal_uses_only_first_k():
    # Two players, 2024. With k=1, only each player's earliest PA counts.
    f = build_pa_frame(_pitches()).filter(pl.col("season") == 2024)
    # player1 earliest (04-01) value .1 ; player2 earliest (04-01) value .8
    mu_k1 = season_mu_causal(f, season=2024, k=1)
    assert abs(mu_k1 - (0.1 + 0.8) / 2) < 1e-12
    # full-season (k huge) = pooled league rate over all PAs
    mu_full = season_mu_causal(f, season=2024, k=10_000)
    all_v = f["value"].to_numpy(); all_d = f["denom"].to_numpy()
    assert abs(mu_full - all_v.sum() / all_d.sum()) < 1e-12


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


def test_posterior_matches_brute_force_gaussian():
    se2, su2 = 0.030 ** 2, 0.015 ** 2
    # prior seasons: two measurements; current: one truncated measurement
    zs = np.array([0.360, 0.330, 0.345]); mus = np.array([0.315, 0.318, 0.320])
    Ss = np.array([0.020 ** 2, 0.022 ** 2, 0.045 ** 2])   # current (last) is noisier
    is_current = np.array([False, False, True])
    theta, V = cutpoint_posterior(zs, mus, Ss, is_current, se2, su2)

    # brute force: latent x=(eta,u0,u1,u_t); build joint, condition on y=z-mu
    P = np.diag([se2, su2, su2, su2])
    H = np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1]], float)
    R = np.diag(Ss); y = zs - mus
    Kf = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    xhat = Kf @ y; Vx = P - Kf @ H @ P
    sel = np.array([1,0,0,1], float)   # eta + u_t
    assert abs(theta - (mus[-1] + sel @ xhat)) < 1e-10
    assert abs(V - sel @ Vx @ sel) < 1e-10

def test_posterior_requires_exactly_one_current():
    se2, su2 = 0.03 ** 2, 0.015 ** 2
    z = np.array([0.34, 0.33]); mu = np.array([0.315, 0.318]); S = np.array([0.02**2, 0.02**2])
    with pytest.raises(AssertionError):
        cutpoint_posterior(z, mu, S, np.array([True, True]), se2, su2)


def test_posterior_no_history_reduces_to_1d_shrinkage():
    # single current measurement, no prior seasons → Phase-1/L2 1-D shrink toward mu_t
    se2, su2 = 0.030 ** 2, 0.015 ** 2
    z, mu, S = np.array([0.400]), np.array([0.315]), np.array([0.050 ** 2])
    theta, V = cutpoint_posterior(z, mu, S, np.array([True]), se2, su2)
    tau2 = se2 + su2                       # prior var of (eta+u_t) with no history
    rel = tau2 / (tau2 + S[0])
    assert abs(theta - (mu[0] + rel * (z[0] - mu[0]))) < 1e-10
    assert abs(V - rel * S[0]) < 1e-10
