"""BBE filtering, outcome classes, feature matrix, non-BBE PA table, stratified subsample."""
from __future__ import annotations

import numpy as np
import polars as pl

HIT_CLASS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
CLASS_NAMES = ["out", "single", "double", "triple", "home_run"]
K = 5
FEATURES_V0 = ["launch_speed", "launch_angle", "sprint_speed"]
FEATURES_SPRAY = ["launch_speed", "launch_angle", "spray_pull", "stand_R", "sprint_speed"]
FEATURES = FEATURES_V0          # back-compat default; run_v0 --variant selects
VARIANT_FEATURES = {"v0": FEATURES_V0, "spray": FEATURES_SPRAY}


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


def build_features(bbe: pl.DataFrame, features: list[str] | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix, float64; no standardization (BART does not need it). `features`
    defaults to the v0 three so existing callers and the frozen v0 path are unchanged."""
    cols = features or FEATURES_V0
    X = bbe.select(cols).to_numpy().astype(np.float64)
    y = bbe["outcome_class"].to_numpy().astype(np.int64)
    return X, y


# Statcast hit-coordinate origin (home plate). hc_x increases toward RIGHT field;
# hc_y DECREASES going out toward the outfield, hence the (198.27 - hc_y) term.
HOME_PLATE_X = 125.42
HOME_PLATE_Y = 198.27
SPRAY_EV_BIN = 5.0        # mph, for the imputation lookup
SPRAY_LA_BIN = 10.0       # degrees
SPRAY_MIN_CELL = 25       # rows required before an (ev, la, stand) cell is trusted


def _spray_cols(bbe: pl.DataFrame) -> pl.DataFrame:
    """phi_raw (RAW direction, degrees: negative = left field, positive = right),
    stand_R (1.0 / 0.0), the observed pull-relative angle, and the lookup bins.

    A right-handed batter PULLS to left field, so the pull-relative angle negates
    phi_raw for stand == 'R' and leaves stand == 'L' alone. Verified empirically on
    2022-25: league mean pull is positive for BOTH hands (L +6.8..+7.5, R +3.2..+3.6)
    and home runs sit at +16..+20 for both, ~80% on the pull side."""
    assert bbe["stand"].null_count() == 0, "stand must be non-null (it is per-EVENT)"
    return bbe.with_columns(
        stand_R=(pl.col("stand") == "R").cast(pl.Float64),
        phi_raw=pl.arctan2(pl.col("hc_x") - HOME_PLATE_X,
                           HOME_PLATE_Y - pl.col("hc_y")).degrees(),
    ).with_columns(
        spray_obs=pl.when(pl.col("stand") == "R").then(-pl.col("phi_raw"))
                    .otherwise(pl.col("phi_raw")),
        _ev_bin=(pl.col("launch_speed") // SPRAY_EV_BIN).cast(pl.Int32),
        _la_bin=(pl.col("launch_angle") // SPRAY_LA_BIN).cast(pl.Int32),
    )


def spray_impute_table(bbe: pl.DataFrame) -> tuple[pl.DataFrame, dict[float, float]]:
    """Median pull-relative spray by (ev_bin, la_bin, stand_R), plus a per-hand
    fallback. Build this on the TRAINING seasons only and apply it to both train and
    holdout, so the holdout never imputes itself."""
    d = _spray_cols(bbe).drop_nulls("spray_obs")
    cell = (
        d.group_by("_ev_bin", "_la_bin", "stand_R")
        .agg(spray_cell=pl.col("spray_obs").median(), _n=pl.len())
        .filter(pl.col("_n") >= SPRAY_MIN_CELL)
        .drop("_n")
        .sort("_ev_bin", "_la_bin", "stand_R")     # polars group_by is not order-stable
    )
    hand = {float(k): float(v) for k, v in
            d.group_by("stand_R").agg(m=pl.col("spray_obs").median()).iter_rows()}
    return cell, hand


def add_spray(bbe: pl.DataFrame, cell: pl.DataFrame,
              hand: dict[float, float]) -> pl.DataFrame:
    """Add phi_raw, spray_pull (POSITIVE = pulled, both hands), stand_R, hc_imputed.

    Rows with null hc_x/hc_y (0.034-0.043% of BBE, 2022-25) are IMPUTED, never dropped
    -- dropping would move the holdout event count off 122,006 and make the -80107 ELPD
    anchor incomparable. Fallback ladder: (ev, la, stand) cell median -> per-hand median
    -> 0.0. hc_imputed is an AUDIT column, deliberately not a BART feature: at ~45 rows
    a season a flag feature is split noise, and five features is the design's spec."""
    d = _spray_cols(bbe).join(cell, on=["_ev_bin", "_la_bin", "stand_R"], how="left")
    fallback = (pl.when(pl.col("stand_R") == 1.0).then(pl.lit(hand.get(1.0, 0.0)))
                  .otherwise(pl.lit(hand.get(0.0, 0.0))))
    out = d.with_columns(
        hc_imputed=pl.col("spray_obs").is_null(),
        spray_pull=pl.coalesce(pl.col("spray_obs"), pl.col("spray_cell"), fallback),
    ).drop("spray_obs", "spray_cell", "_ev_bin", "_la_bin")
    assert out["spray_pull"].null_count() == 0, "spray_pull must be fully imputed"
    return out


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


def stratified_subsample(df: pl.DataFrame, n: int | None, seed: int) -> pl.DataFrame:
    """Proportional (largest-remainder) subsample by outcome_class. Preserves class
    proportions so triples survive; NEVER rebalances (calibration is a gate — the
    mlb-hit-classifier resampling study showed rebalancing distorts probabilities)."""
    if n is None or n >= df.height:
        return df
    counts = dict(df.group_by("outcome_class").len().iter_rows())
    shares = {c: cnt * n / df.height for c, cnt in counts.items()}
    alloc = {c: int(s) for c, s in shares.items()}
    remainder = n - sum(alloc.values())
    # Secondary key = class index so ties break deterministically: polars group_by is
    # NOT order-stable, so a tie on the fractional part must not depend on row order.
    for c, _ in sorted(shares.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))[:remainder]:
        alloc[c] += 1
    alloc = {c: min(k, counts[c]) for c, k in alloc.items()}
    parts = [
        df.filter(pl.col("outcome_class") == c).sample(n=k, seed=seed + int(c), shuffle=True)
        for c, k in sorted(alloc.items())
        if k > 0
    ]
    return pl.concat(parts).sample(fraction=1.0, seed=seed, shuffle=True)
