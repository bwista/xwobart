import numpy as np
import polars as pl

from src.talent2 import (
    FLOOR_SD_PER_PA,
    assemble_measurements,
    bootstrap_S,
    build_pa_measurements,
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
