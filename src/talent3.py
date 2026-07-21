"""Rest-of-season xwOBA forecast: a hierarchical player-talent model evaluated at
mid-season cutpoints (spec: docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-
forecast-design.md). Rung a: career random intercept + iid season drift, xwOBA-only,
no aging. Reuses talent2's bootstrap_S for the measurement variance. Pure functions;
orchestration in scripts/run_talent3.py."""
from __future__ import annotations

import numpy as np
import polars as pl

from src.talent2 import FLOOR_SD_PER_PA, bootstrap_S
from scipy.optimize import minimize


def build_pa_frame(pitches: pl.DataFrame) -> pl.DataFrame:
    """One row per PA with (batter, season, game_date, value, denom). value/denom
    exactly as talent2.build_pa_measurements; game_date is parsed to pl.Date so
    PAs can be ordered within a season for cutpoints."""
    return (
        pitches.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value")),
            game_date=pl.col("game_date").cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d"),
        )
        .select("batter", season="game_year", game_date="game_date",
                value="value", denom="woba_denom")
    )


def cutpoint_split(pas: pl.DataFrame, k: int, min_remaining: float) -> dict | None:
    """Order one player-season's PAs by game_date, split first-k (observed) vs rest
    (remaining). Returns a dict with r_obs/D_obs, r_rest/D_rest (realized), w, and
    the raw observed/remaining value+denom arrays. None if fewer than k+1 PAs or the
    remaining denom < min_remaining (no real 'rest of season')."""
    o = pas.sort("game_date", maintain_order=True)   # stable: ties keep input order
    n = o.height
    if n <= k:
        return None
    v = o["value"].to_numpy(); d = o["denom"].to_numpy()
    v_obs, d_obs = v[:k], d[:k]
    v_rest, d_rest = v[k:], d[k:]
    D_obs, D_rest = float(d_obs.sum()), float(d_rest.sum())
    if D_rest < min_remaining or D_obs <= 0:
        return None
    return {
        "r_obs": float(v_obs.sum() / D_obs), "D_obs": D_obs,
        "r_rest": float(v_rest.sum() / D_rest), "D_rest": D_rest,
        "w": D_rest / (D_obs + D_rest),
        "v_obs": v_obs, "d_obs": d_obs, "v_rest": v_rest, "d_rest": d_rest,
    }


def sample_measurement(value: np.ndarray, denom: np.ndarray, B: int,
                       rng: np.random.Generator) -> tuple[float, float]:
    """Measurement (rate) and its floored bootstrap sampling variance for one PA
    sample. xwOBA-only: reuse bootstrap_S with NaN peripherals and take S[0,0]."""
    nan = np.full(len(value), np.nan)
    S = bootstrap_S(value, denom, nan, nan, B=B, rng=rng)
    z = float(value.sum() / denom.sum())
    return z, float(S[0, 0])


def season_mu_causal(pa_frame: pl.DataFrame, season: int, k: int) -> float:
    """League environment for `season`: pooled rate over every player's first-k PAs
    (ordered by game_date). Causal — uses no PA beyond a cutpoint. k huge → full season."""
    s = (
        pa_frame.filter(pl.col("season") == season)
        .sort("game_date", maintain_order=True)
        .with_columns(_rank=pl.col("game_date").cum_count().over("batter"))
        .filter(pl.col("_rank") <= k)
    )
    return float(s["value"].sum() / s["denom"].sum())


def fit_hypers_eb(y: np.ndarray, S: np.ndarray, player_idx: np.ndarray
                  ) -> tuple[float, float]:
    """Marginal-MLE of (sigma_eta^2, sigma_u^2) for the rung-a hierarchy.
    y = z - mu_s (demeaned measurements), S = per-obs measurement variance,
    player_idx groups rows by player. Per player: y_i ~ N(0, sig_eta^2 11' +
    diag(sig_u^2 + S_i)). Optimizes log-variances (positivity) with L-BFGS-B."""
    groups = [np.where(player_idx == p)[0] for p in np.unique(player_idx)]

    def nll(theta: np.ndarray) -> float:
        se2, su2 = np.exp(theta)
        total = 0.0
        for g in groups:
            yi, Si = y[g], S[g]
            Sig = se2 * np.ones((len(g), len(g))) + np.diag(su2 + Si)
            sign, logdet = np.linalg.slogdet(Sig)
            sol = np.linalg.solve(Sig, yi)
            total += 0.5 * (logdet + yi @ sol)
        return float(total)

    res = minimize(nll, np.log([0.02 ** 2, 0.01 ** 2]),
                   method="L-BFGS-B", options={"maxiter": 200})
    se2, su2 = np.exp(res.x)
    return float(se2), float(su2)
