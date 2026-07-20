"""Level-2 true-talent xwOBA: a joint MVN measurement model over (raw xwOBA,
average exit velocity, barrel rate) per batter-season
(spec: docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md).

Phase 1 (src/talent.py) shrinks raw xwOBA toward the season league mean. Level 2
upgrades the prior: the three stats are jointly noisy measurements of correlated
latent talents, z_i ~ N((theta_i, xi_i), S_i), (theta_i, xi_i) ~ N(mu_season,
Sigma_talent). The per-player measurement covariance S_i is bootstrapped from the
player's own PAs; its OFF-DIAGONALS carry the shared sampling noise (all three
stats come from the same balls) — modeling them explicitly is what keeps the
low-PA gains honest. Posterior E[theta|z] leans on the fast-stabilizing
peripherals exactly when the xwOBA sample is small, and reduces to Phase 1 when
the peripheral dims are dropped. Pure functions only — orchestration lives in
scripts/run_talent2.py."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize

Z90 = 1.6448536269514722          # 90% two-sided normal quantile (as src/talent.py)
DIMS = ("xwoba", "avg_ev", "barrel_rate")
MIN_BBE = 5                       # fewer tracked BBE -> 1-D fallback (peripherals carry ~nothing)
FLOOR_SD_PER_PA = 0.25            # xwOBA meas-variance floor = (0.25)^2/n  (NOTES.md limitation 3)


def build_pa_measurements(pitches: pl.DataFrame) -> pl.DataFrame:
    """One row per plate appearance: (batter, season, value, denom, ev, barrel).
    value/denom exactly as talent.build_pa_values; ev = launch_speed and
    barrel = (launch_speed_angle == 6) only on tracked BBE (type X with non-null
    launch_speed AND launch_speed_angle — the ~0.3% untracked BBE keep their
    xwOBA value but are excluded from the peripheral denominators)."""
    tracked = (
        (pl.col("type") == "X")
        & pl.col("launch_speed").is_not_null()
        & pl.col("launch_speed_angle").is_not_null()
    )
    return (
        pitches.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value")),
            ev=pl.when(tracked).then(pl.col("launch_speed")),
            barrel=pl.when(tracked).then(
                (pl.col("launch_speed_angle") == 6).cast(pl.Float64)
            ),
        )
        .select("batter", season="game_year", value="value", denom="woba_denom",
                ev="ev", barrel="barrel")
    )
