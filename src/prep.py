"""BBE filtering, outcome classes, feature matrix, non-BBE PA table, stratified subsample."""
from __future__ import annotations

import numpy as np
import polars as pl

HIT_CLASS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
CLASS_NAMES = ["out", "single", "double", "triple", "home_run"]
K = 5
FEATURES = ["launch_speed", "launch_angle", "sprint_speed"]


def filter_bbe(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """type == 'X', minus bunts (via `des` text — `description` never catches in-play
    bunts, spec §6.1), minus rows missing launch_speed/launch_angle.
    Returns (bbe, per-season drop report)."""
    x = df.filter(pl.col("type") == "X").with_columns(
        is_bunt=pl.col("des").fill_null("").str.to_lowercase().str.contains("bunt"),
        missing_ls_la=pl.col("launch_speed").is_null() | pl.col("launch_angle").is_null(),
    )
    report = (
        x.group_by("game_year")
        .agg(
            n_bbe_raw=pl.len(),
            n_bunt=pl.col("is_bunt").sum(),
            n_missing_ls_la=(pl.col("missing_ls_la") & ~pl.col("is_bunt")).sum(),
        )
        .with_columns(
            pct_missing=(100 * pl.col("n_missing_ls_la")
                         / (pl.col("n_bbe_raw") - pl.col("n_bunt"))).round(2)
        )
        .sort("game_year")
    )
    bbe = x.filter(~pl.col("is_bunt") & ~pl.col("missing_ls_la")).drop("is_bunt", "missing_ls_la")
    return bbe, report


def add_outcome_class(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        outcome_class=pl.col("events").replace_strict(HIT_CLASS, default=0, return_dtype=pl.Int8)
    )


def class_distribution(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("outcome_class").len().sort("outcome_class")
        .with_columns(pct=(100 * pl.col("len") / df.height))
    )


def build_features(bbe: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exactly three features, float64; no standardization (BART does not need it)."""
    X = bbe.select(FEATURES).to_numpy().astype(np.float64)
    y = bbe["outcome_class"].to_numpy().astype(np.int64)
    return X, y


def build_non_bbe_pa(df: pl.DataFrame) -> pl.DataFrame:
    """Walks, HBP, strikeouts, catcher's interference: woba_denom == 1 and type != 'X'."""
    return (
        df.filter((pl.col("woba_denom") == 1) & (pl.col("type") != "X"))
        .select(
            "batter",
            season=pl.col("game_year"),
            woba_value=pl.col("woba_value"),
            woba_denom=pl.col("woba_denom"),
        )
    )
