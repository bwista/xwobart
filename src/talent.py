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
