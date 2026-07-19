"""True-talent xwOBA via empirical-Bayes shrinkage (spec: 2026-07-18 talent plan).

Each player-season's raw xwOBA is a noisy estimate of their true talent; shrink it
toward the season population by its reliability (a function of sample size), giving a
center that regresses small samples toward the mean and an interval that narrows with
PA. Pure functions only — orchestration lives in scripts/run_talent.py."""
from __future__ import annotations

import numpy as np
import polars as pl

Z90 = 1.6448536269514722  # 90% two-sided normal quantile


def build_pa_values(pitches: pl.DataFrame) -> pl.DataFrame:
    """One row per plate appearance: (batter, season, value, denom). A PA's xwOBA
    value is the batted-ball estimate for BBE (est_woba, falling back to the
    deterministic woba_value if missing) and woba_value for walks/Ks/HBP. Only
    PA-ending rows (woba_denom not null) count."""
    return (
        pitches.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value"))
        )
        .select("batter", season="game_year", value="value", denom="woba_denom")
    )


def per_player_raw(pav: pl.DataFrame) -> pl.DataFrame:
    """Per (batter, season): raw xwOBA = Σvalue/Σdenom, PA, and the sampling SE of
    that mean (sd of per-PA values / √n). se2 is the variance used by the EB fit."""
    return (
        pav.group_by("batter", "season")
        .agg(
            PA=pl.col("denom").sum().cast(pl.Int64),
            n=pl.len(),
            num=pl.col("value").sum(),
            den=pl.col("denom").sum(),
            sd=pl.col("value").std(ddof=1),
        )
        .with_columns(
            xwoba_raw=pl.col("num") / pl.col("den"),
            se=pl.col("sd") / pl.col("n").sqrt(),
        )
        .with_columns(se2=pl.col("se") ** 2)
        .drop("num", "den", "sd")
        .sort("batter", "season")
    )


def eb_fit(raw: np.ndarray, se2: np.ndarray, tau2_floor: float = 1e-8) -> tuple[float, float]:
    """Gaussian–Gaussian empirical Bayes by method of moments (DerSimonian–Laird
    style). Observed variance of raw = between-player τ² + mean within-player SE²;
    μ is the precision-weighted mean. Returns (mu, tau2)."""
    raw = np.asarray(raw, float)
    se2 = np.asarray(se2, float)
    grand = raw.mean()
    tau2 = max(float(((raw - grand) ** 2).mean() - se2.mean()), tau2_floor)
    w = 1.0 / (tau2 + se2)
    mu = float((w * raw).sum() / w.sum())
    # one refinement of tau2 around the weighted mean
    tau2 = max(float((w * ((raw - mu) ** 2 - se2)).sum() / w.sum()), tau2_floor)
    return mu, tau2


def eb_shrink(raw: np.ndarray, se2: np.ndarray, mu: float, tau2: float):
    """Posterior for each player's true talent under N(mu, tau2) prior and N(theta, se2)
    likelihood. Returns (theta_hat, post_var, ci_lo, ci_hi, reliability)."""
    raw = np.asarray(raw, float)
    se2 = np.asarray(se2, float)
    rel = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    theta = mu + rel * (raw - mu)
    post_var = rel * se2                      # = tau2*se2/(tau2+se2); 0 when tau2==0
    half = Z90 * np.sqrt(post_var)
    return theta, post_var, theta - half, theta + half, rel


def build_talent_table(pav: pl.DataFrame, fit_min_pa: int = 100) -> pl.DataFrame:
    """Per (batter, season): raw xwOBA, EB true-talent estimate, 90% interval, and
    reliability. EB hyperparameters (mu, tau2) are fit per season on players with
    PA >= fit_min_pa, then applied to all player-seasons. Single-PA seasons have no
    sampling SD (se2 null) and are dropped — they carry no estimable uncertainty."""
    raw = per_player_raw(pav).filter(pl.col("se2").is_not_null())
    out = []
    for season in raw["season"].unique().sort().to_list():
        s = raw.filter(pl.col("season") == season)
        fit = s.filter(pl.col("PA") >= fit_min_pa)
        mu, tau2 = eb_fit(fit["xwoba_raw"].to_numpy(), fit["se2"].to_numpy())
        theta, pv, lo, hi, rel = eb_shrink(
            s["xwoba_raw"].to_numpy(), s["se2"].to_numpy(), mu, tau2
        )
        out.append(s.with_columns(
            xwoba_talent=pl.Series(theta),
            talent_post_var=pl.Series(pv),
            talent_lo=pl.Series(lo),
            talent_hi=pl.Series(hi),
            reliability=pl.Series(rel),
            mu_season=pl.lit(mu),
            tau_season=pl.lit(float(np.sqrt(tau2))),
        ))
    return pl.concat(out).sort("batter", "season")
