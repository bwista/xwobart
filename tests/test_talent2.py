import numpy as np
import polars as pl

from src.talent import build_pa_values, build_talent_table, eb_fit, eb_shrink
from src.talent2 import (
    FLOOR_SD_PER_PA,
    MIN_BBE,
    assemble_measurements,
    bootstrap_S,
    build_pa_measurements,
    build_talent2_table,
    mvn_mle,
    mvn_posterior,
    player_measurements,
)


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


def _rng():
    return np.random.default_rng(7)


def test_player_measurements_triple_and_dropped_singleton():
    pam = build_pa_measurements(_pitches())
    # add a single-PA player (no sd -> se2 null -> dropped, like Phase 1)
    single = pl.DataFrame({"batter": [9], "season": [2024], "value": [0.3],
                           "denom": [1], "ev": [90.0], "barrel": [0.0]})
    meas = player_measurements(pl.concat([pam, single.cast(pam.schema)]))
    assert 9 not in meas["batter"].to_list()
    r1 = meas.filter(pl.col("batter") == 1).row(0, named=True)
    assert r1["n"] == 3 and r1["n_bbe"] == 2
    assert abs(r1["avg_ev"] - (101.3 + 88.0) / 2) < 1e-9
    assert abs(r1["barrel_rate"] - 0.5) < 1e-9
    assert abs(r1["xwoba_raw"] - (1.2 + 0.1 + 0.69) / 3) < 1e-9
    assert r1["se2"] > 0
    # player 2: BBE untracked -> n_bbe == 0, peripherals null
    r2 = meas.filter(pl.col("batter") == 2).row(0, named=True)
    assert r2["n_bbe"] == 0 and r2["avg_ev"] is None


def test_bootstrap_S_recovers_known_covariance():
    # Player with n=4000 iid PAs where value and ev share noise by construction.
    rng = _rng()
    n = 4000
    common = rng.normal(0, 1.0, n)
    ev = 89.0 + 5.0 * common + rng.normal(0, 3.0, n)
    value = np.clip(0.32 + 0.20 * common + rng.normal(0, 0.40, n), 0, 2.0)
    barrel = (rng.random(n) < 0.05 + 0.04 * (common > 1)).astype(float)
    denom = np.ones(n)
    S = bootstrap_S(value, denom, ev, barrel, B=800, rng=_rng())
    # diagonal ~ Var(stat of the mean): Var(value)/n etc. (within 25% — B noise)
    assert abs(S[0, 0] / (value.var(ddof=1) / n) - 1) < 0.25
    assert abs(S[1, 1] / (ev.var(ddof=1) / n) - 1) < 0.25
    # shared noise -> positive xwOBA/EV cross-covariance, right magnitude
    expected_cov = np.cov(value, ev)[0, 1] / n
    assert S[0, 1] > 0 and abs(S[0, 1] / expected_cov - 1) < 0.35


def test_bootstrap_S_xwoba_var_matches_analytic_se2():
    rng = _rng()
    n = 500
    value = rng.normal(0.32, 0.45, n)
    S = bootstrap_S(value, np.ones(n), np.full(n, np.nan), np.full(n, np.nan),
                    B=800, rng=_rng())
    # peripherals absent -> those entries NaN, xwOBA var still valid
    assert np.isnan(S[1, 1]) and np.isnan(S[2, 2])
    analytic = value.var(ddof=1) / n
    assert abs(S[0, 0] / analytic - 1) < 0.2


def test_bootstrap_S_floor_on_degenerate_values():
    # identical values -> zero bootstrap variance -> floored, not 0 (NOTES lim. 3)
    n = 8
    S = bootstrap_S(np.full(n, 0.7), np.ones(n), np.full(n, np.nan),
                    np.full(n, np.nan), B=200, rng=_rng())
    assert S[0, 0] >= FLOOR_SD_PER_PA ** 2 / n


def test_bootstrap_S_3channel_unchanged():
    # BIT-IDENTITY guard: baseline captured from the pre-change bootstrap_S on
    # this exact fixture (B=400, seed 0) via a one-off script. pull=None must
    # reproduce it exactly -- proves the k-channel generalization doesn't
    # perturb the RNG draw or the 3-channel arithmetic at all.
    v = np.array([0.5, 1.2, 0.0, 2.0, 0.3]); d = np.ones(5)
    ev = np.array([90., 101., 88., 104., np.nan]); br = np.array([0., 1., 0., 1., np.nan])
    S = bootstrap_S(v, d, ev, br, B=400, rng=np.random.default_rng(0))
    S_baseline = np.array([
        [0.11355162907268171, 1.1190691938178783, 0.07629323308270675],
        [1.1190691938178783, 13.963786255569477, 0.9978526176552491],
        [0.07629323308270675, 0.9978526176552491, 0.07358535226956278],
    ])
    assert S.shape == (3, 3)
    assert np.array_equal(S, S_baseline)           # bit-identical, not just finite


def test_bootstrap_S_4channel_pull():
    v = np.array([0.5, 1.2, 0.0, 2.0, 0.3]); d = np.ones(5)
    ev = np.array([90., 101., 88., 104., np.nan]); br = np.array([0., 1., 0., 1., np.nan])
    pull = np.array([-5., 18., 2., 22., np.nan])
    S = bootstrap_S(v, d, ev, br, B=400, rng=np.random.default_rng(0), pull=pull)
    assert S.shape == (4, 4)
    w = np.linalg.eigvalsh(S)
    assert (w >= -1e-9).all()          # PSD
    # pull is an ADDITIVE channel: the other three are undisturbed. NOT bit-exact --
    # np.cov's BLAS accumulation differs ~1 ULP across 3-row vs 4-row input.
    S3 = bootstrap_S(v, d, ev, br, B=400, rng=np.random.default_rng(0))
    assert np.allclose(S[:3, :3], S3, atol=1e-12)
    assert S[3, 3] >= 1e-8             # 4th (pull) diagonal got floored by range(1, D)
    assert np.array_equal(S, S.T)      # symmetric


def test_assemble_measurements_aligns_and_flags():
    rng = _rng()
    rows = []
    for batter, n_pa in ((1, 200), (2, 40), (3, 3)):
        for _ in range(n_pa):
            bbe = rng.random() < 0.7
            rows.append({
                "batter": batter, "game_year": 2024, "type": "X" if bbe else "S",
                "launch_speed": float(rng.normal(89, 6)) if bbe else None,
                "launch_speed_angle": float(rng.integers(1, 7)) if bbe else None,
                "estimated_woba_using_speedangle": float(np.clip(rng.normal(0.35, 0.4), 0, 2)) if bbe else None,
                "woba_value": 0.0, "woba_denom": 1,
            })
    meas, S = assemble_measurements(build_pa_measurements(pl.DataFrame(rows)),
                                    B=200, seed=1)
    assert S.shape == (meas.height, 3, 3) and meas["s_ok"].all()
    # rows are sorted (batter, season) and S is row-aligned: bigger sample ->
    # smaller xwOBA measurement variance
    m = {b: S[i, 0, 0] for i, b in enumerate(meas["batter"].to_list())}
    assert m[1] < m[2] < m[3]


# n is set so the recovery tolerances below are ~4 sampling SEs, not ~2: with
# Var(z) = Sigma + E[S] ~ 1.9 per dim, SE(mu_season) ~ sqrt(1.9/(n/2)), so n=8000
# gives SE ~ 0.022 against the 0.08 tolerance. (At n=3000 the seed-3 draw lands
# 2.5 SEs out and the test fails for reasons that have nothing to do with the code.)
def _simulate_mvn(n=8000, seed=3):
    rng = np.random.default_rng(seed)
    mu_true = np.array([[0.0, 0.2, -0.1], [0.3, -0.2, 0.1]])   # 2 seasons x 3 dims
    sd = np.array([1.0, 0.8, 0.6])
    C = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]])
    Sigma_true = C * np.outer(sd, sd)
    t = rng.integers(0, 2, n)
    theta = mu_true[t] + rng.multivariate_normal(np.zeros(3), Sigma_true, n)
    S = np.empty((n, 3, 3))
    z = np.empty((n, 3))
    for i in range(n):
        s_sd = rng.uniform(0.3, 1.5, 3)
        Si = np.diag(s_sd ** 2)
        Si[0, 1] = Si[1, 0] = 0.5 * s_sd[0] * s_sd[1]       # shared noise
        S[i] = Si
        z[i] = theta[i] + rng.multivariate_normal(np.zeros(3), Si)
    return z, S, t, mu_true, Sigma_true


def test_mvn_mle_parameter_recovery():
    z, S, t, mu_true, Sigma_true = _simulate_mvn()
    mu, Sigma = mvn_mle(z, S, t, n_seasons=2)
    assert np.abs(mu - mu_true).max() < 0.08
    assert np.abs(np.diag(Sigma) / np.diag(Sigma_true) - 1).max() < 0.15
    corr = Sigma / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))
    corr_true = Sigma_true / np.sqrt(np.outer(np.diag(Sigma_true), np.diag(Sigma_true)))
    assert np.abs(corr - corr_true).max() < 0.12


def test_mvn_mle_1d_matches_eb_fit():
    # Same setup as Phase 1's test_eb_fit_recovers_hyperparameters
    rng = np.random.default_rng(0)
    n = 4000
    theta = rng.normal(0.32, 0.05, n)
    se = rng.uniform(0.02, 0.08, n)
    raw = theta + rng.normal(0, se)
    mu, Sigma = mvn_mle(raw[:, None], (se ** 2)[:, None, None],
                        np.zeros(n, int), n_seasons=1)
    mu_eb, tau2_eb = eb_fit(raw, se ** 2)
    assert abs(mu[0, 0] - mu_eb) < 0.003
    assert abs(np.sqrt(Sigma[0, 0]) - np.sqrt(tau2_eb)) < 0.005


def test_mvn_posterior_1d_equals_eb_shrink():
    mu_s, tau2 = 0.320, 0.05 ** 2
    raw = np.array([0.40, 0.40, 0.25])
    se2 = np.array([0.02 ** 2, 0.08 ** 2, 0.03 ** 2])
    theta, var0 = mvn_posterior(raw[:, None], se2[:, None, None],
                                np.array([[mu_s]]), np.array([[tau2]]),
                                np.zeros(3, int))
    t_eb, pv_eb, *_ = eb_shrink(raw, se2, mu_s, tau2)
    assert np.allclose(theta[:, 0], t_eb) and np.allclose(var0, pv_eb)


def test_mvn_posterior_peripheral_pull_and_information_gain():
    # Two low-PA players, identical league-average xwOBA measurement; one has
    # elite, precisely-measured peripherals. Positive talent correlations must
    # pull his xwOBA talent above the mean, with LOWER posterior variance than
    # the 1-D shrink of the same xwOBA measurement.
    mu = np.array([[0.0, 0.0, 0.0]])
    C = np.array([[1.0, 0.7, 0.6], [0.7, 1.0, 0.4], [0.6, 0.4, 1.0]])
    z = np.array([[0.0, 2.0, 2.0],     # league xwOBA, elite peripherals
                  [0.0, 0.0, 0.0]])    # league everything
    S = np.tile(np.diag([4.0, 0.1, 0.1]), (2, 1, 1))   # xwOBA noisy, periphs tight
    theta, var0 = mvn_posterior(z, S, mu, C, np.zeros(2, int))
    assert theta[0, 0] > 0.5 and abs(theta[1, 0]) < 1e-9
    t1, v1 = mvn_posterior(z[:, :1], S[:, :1, :1], mu[:, :1], C[:1, :1],
                           np.zeros(2, int))
    assert var0[0] < v1[0]             # peripherals resolve xwOBA-talent variance


def _synthetic_league(n_players=150, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for batter in range(n_players):
        talent_ev = rng.normal(89, 3)
        talent_x = 0.31 + 0.012 * (talent_ev - 89) + rng.normal(0, 0.02)
        n_pa = int(rng.integers(4, 500))
        # Guarantee the 1-D fallback is exercised rather than left to the draw:
        # at n_pa <= 4, n_bbe < MIN_BBE (=5) by construction, whatever the RNG does.
        if batter < 2:
            n_pa = 4

        for _ in range(n_pa):
            bbe = rng.random() < 0.7
            ev = rng.normal(talent_ev, 7) if bbe else None
            val = float(np.clip(rng.normal(talent_x, 0.45), 0, 2)) if bbe else \
                  float(rng.random() < 0.3) * 0.7
            rows.append({
                "batter": batter, "game_year": 2024, "type": "X" if bbe else "S",
                "launch_speed": ev,
                "launch_speed_angle": (6.0 if (bbe and ev > 99) else 3.0) if bbe else None,
                "estimated_woba_using_speedangle": val if bbe else None,
                "woba_value": val, "woba_denom": 1,
            })
    return pl.DataFrame(rows)


def test_build_talent2_table_structure_and_fallback():
    pitches = _synthetic_league()
    pam = build_pa_measurements(pitches)
    meas, S = assemble_measurements(pam, B=200, seed=2)
    tbl, hypers = build_talent2_table(meas, S, fit_min_pa=100)
    need = {"batter", "season", "PA", "n_bbe", "xwoba_raw", "avg_ev",
            "barrel_rate", "xwoba_talent2", "talent2_var", "talent2_lo",
            "talent2_hi", "reliability2", "used_dims"}
    assert need.issubset(tbl.columns) and tbl.height == meas.height
    assert set(tbl["used_dims"].unique().to_list()) <= {"3d", "1d"}
    # the fallback path must actually be exercised, not vacuously true
    tiny = tbl.filter(pl.col("n_bbe") < MIN_BBE)
    assert tiny.height > 0 and tiny["used_dims"].eq("1d").all()
    assert tbl["talent2_lo"].lt(tbl["talent2_hi"]).all()
    assert 0.25 < hypers["mu"][0][0] < 0.40          # xwOBA league mean, unstd
    # positive xwOBA/EV talent correlation was built in -> recovered sign
    assert hypers["Sigma"][0][1] > 0


def test_build_talent2_table_1d_matches_phase1():
    pitches = _synthetic_league()
    pam = build_pa_measurements(pitches)
    meas, S = assemble_measurements(pam, B=400, seed=2)
    tbl, _ = build_talent2_table(meas, S, dims=("xwoba",), fit_min_pa=100)
    p1 = build_talent_table(build_pa_values(pitches), fit_min_pa=100)
    j = tbl.join(p1.select("batter", "season", "xwoba_talent"),
                 on=["batter", "season"], how="inner")
    assert j.height == tbl.height
    d = (j["xwoba_talent2"] - j["xwoba_talent"]).abs()
    r = np.corrcoef(j["xwoba_talent2"].to_numpy(), j["xwoba_talent"].to_numpy())[0, 1]
    assert r > 0.995 and d.median() < 0.005


def test_build_talent2_table_peripheral_pull_at_low_pa():
    # In the synthetic league, xwOBA talent tracks EV talent. Among LOW-PA
    # players, the 3-D posterior must move high-EV players up relative to the
    # 1-D (Phase-1-style) shrink of the same xwOBA measurement.
    pitches = _synthetic_league(seed=13)
    pam = build_pa_measurements(pitches)
    meas, S = assemble_measurements(pam, B=200, seed=2)
    t3, _ = build_talent2_table(meas, S, fit_min_pa=100)
    t1, _ = build_talent2_table(meas, S, dims=("xwoba",), fit_min_pa=100)
    j = (t3.select("batter", "PA", "avg_ev", "n_bbe", t2=pl.col("xwoba_talent2"))
           .join(t1.select("batter", t1c=pl.col("xwoba_talent2")), on="batter")
           .filter((pl.col("PA") < 80) & (pl.col("n_bbe") >= 5)))
    hi = j.filter(pl.col("avg_ev") > j["avg_ev"].quantile(0.8))
    lo = j.filter(pl.col("avg_ev") < j["avg_ev"].quantile(0.2))
    assert (hi["t2"] - hi["t1c"]).mean() > (lo["t2"] - lo["t1c"]).mean()
