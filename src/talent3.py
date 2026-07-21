"""Rest-of-season xwOBA forecast: a hierarchical player-talent model evaluated at
mid-season cutpoints (spec: docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-
forecast-design.md). Rung a: career random intercept + iid season drift, xwOBA-only,
no aging. Reuses talent2's bootstrap_S for the measurement variance. Pure functions;
orchestration in scripts/run_talent3.py."""
from __future__ import annotations

import numpy as np
import polars as pl

from src.talent2 import FLOOR_SD_PER_PA, bootstrap_S


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
