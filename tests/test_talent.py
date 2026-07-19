import numpy as np
import polars as pl

from src.talent import (
    build_pa_values,
    build_talent_table,
    eb_fit,
    eb_shrink,
    per_player_raw,
)


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


def test_eb_fit_recovers_hyperparameters():
    # Simulate true talents ~ N(0.32, 0.05^2); noisy observations with known SEs.
    rng = np.random.default_rng(0)
    n = 4000
    theta = rng.normal(0.32, 0.05, n)
    se = rng.uniform(0.02, 0.08, n)            # heteroscedastic (small vs large samples)
    raw = theta + rng.normal(0, se)
    mu, tau2 = eb_fit(raw, se ** 2)
    assert abs(mu - 0.32) < 0.005
    assert abs(np.sqrt(tau2) - 0.05) < 0.01


def test_eb_shrink_monotone_and_centered():
    mu, tau2 = 0.320, 0.05 ** 2
    raw = np.array([0.400, 0.400])             # same raw, different precision
    se2 = np.array([0.02 ** 2, 0.08 ** 2])     # small SE (many PA) vs large SE (few PA)
    theta, pv, lo, hi, rel = eb_shrink(raw, se2, mu, tau2)
    # smaller SE -> higher reliability -> less shrinkage -> theta closer to raw
    assert rel[0] > rel[1]
    assert theta[0] > theta[1]
    assert mu < theta[1] < raw[1]              # shrinks toward mu, never past it
    # posterior interval is narrower for the more reliable (smaller-SE) estimate
    assert (hi[0] - lo[0]) < (hi[1] - lo[1])
    # posterior variance = tau2*se2/(tau2+se2)
    assert np.allclose(pv, tau2 * se2 / (tau2 + se2))


def test_eb_shrink_edge_zero_tau():
    # No between-player spread -> full shrink to mu, degenerate interval floored, not NaN.
    theta, pv, lo, hi, rel = eb_shrink(np.array([0.5]), np.array([0.01]), 0.32, 0.0)
    assert theta[0] == 0.32 and rel[0] == 0.0 and np.isfinite(lo[0]) and np.isfinite(hi[0])


def test_build_talent_table_per_season_and_small_samples_shrink_more():
    rng = np.random.default_rng(1)
    rows = []
    for season, lg in ((2023, 0.33), (2024, 0.31)):
        for batter in range(300):
            pa = int(rng.integers(50, 650))
            theta = rng.normal(lg, 0.045)
            vals = rng.normal(theta, 0.9, pa)            # per-PA values, high variance
            for v in vals:
                rows.append({"batter": batter, "game_year": season,
                             "type": "X", "estimated_woba_using_speedangle": v,
                             "woba_value": v, "woba_denom": 1})
    pav = build_pa_values(pl.DataFrame(rows))
    tbl = build_talent_table(pav, fit_min_pa=100)
    assert {"batter", "season", "PA", "xwoba_raw", "xwoba_talent",
            "talent_lo", "talent_hi", "reliability", "mu_season"}.issubset(tbl.columns)
    # per-season mu differs and is near the simulated league levels
    mus = dict(tbl.group_by("season").agg(pl.col("mu_season").first()).iter_rows())
    assert abs(mus[2023] - 0.33) < 0.02 and abs(mus[2024] - 0.31) < 0.02
    # low-PA players are shrunk harder (reliability rises with PA)
    lo = tbl.filter(pl.col("PA") < 120)["reliability"].mean()
    hi = tbl.filter(pl.col("PA") > 500)["reliability"].mean()
    assert hi > lo
    # talent estimate lies between raw and its season mean (shrinkage direction)
    s = tbl.filter(pl.col("PA") < 120)
    pulled = ((s["xwoba_talent"] - s["mu_season"]).abs()
              <= (s["xwoba_raw"] - s["mu_season"]).abs() + 1e-9).all()
    assert pulled
