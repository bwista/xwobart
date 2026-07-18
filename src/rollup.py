"""Linear weights, per-draw expected values, player-season rollup, player table."""
from __future__ import annotations

import numpy as np
import polars as pl

from src.prep import K

# Sanity windows (spec §8.1). Out is ~0.016 empirically because Savant credits
# field_error / fielders_choice rows at woba_value 0.9 and those map to class 0.
WEIGHT_SANITY = {0: (0.0, 0.05), 1: (0.80, 1.00), 2: (1.10, 1.40), 3: (1.35, 1.80), 4: (1.85, 2.20)}


def linear_weights(train_bbe: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """w_k = mean observed woba_value by outcome class over training-season BBE."""
    agg = (
        train_bbe.drop_nulls("woba_value")
        .group_by("outcome_class").agg(pl.col("woba_value").mean())
        .sort("outcome_class")
    )
    w = np.zeros(K)
    w[agg["outcome_class"].to_numpy().astype(int)] = agg["woba_value"].to_numpy()
    warnings = [
        f"w[{c}]={w[c]:.3f} outside sanity range {rng}"
        for c, rng in WEIGHT_SANITY.items()
        if not (rng[0] <= w[c] <= rng[1])
    ]
    return w, warnings


def expected_values(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """p (S, K, n) class probabilities per draw -> (S, n) expected wOBA values."""
    return np.tensordot(w, p, axes=(0, 1)).astype(np.float32)


def player_rollup(ev_draws: np.ndarray, bbe_keys: pl.DataFrame, non_bbe: pl.DataFrame) -> pl.DataFrame:
    """Per-draw player-season xwOBA matching the public construction (spec §8.3).

    ev_draws (S, n) is row-aligned with bbe_keys (columns: batter, season, woba_denom).
    numerator = sum of expected event values (BBE) + sum of actual woba_value (non-BBE)
    denominator = sum of woba_denom over both. Computed per draw, then summarized.
    """
    S, n = ev_draws.shape
    assert bbe_keys.height == n, "ev_draws rows must align with bbe_keys rows"

    keys = (
        pl.concat([
            bbe_keys.select("batter", "season"),
            non_bbe.select("batter", "season"),
        ])
        .unique()
        .sort("batter", "season")
        .with_row_index("g")
    )
    G = keys.height

    # Joins do not guarantee row order — carry an index and sort back (alignment test above).
    g_bbe = (
        bbe_keys.with_row_index("_i")
        .join(keys, on=["batter", "season"], how="left")
        .sort("_i")["g"].to_numpy().astype(np.int64)
    )
    num = np.zeros((S, G))
    for s in range(S):
        num[s] = np.bincount(g_bbe, weights=ev_draws[s].astype(np.float64), minlength=G)
    den_bbe = np.bincount(g_bbe, weights=bbe_keys["woba_denom"].to_numpy(), minlength=G)

    if non_bbe.height:
        nb = non_bbe.join(keys, on=["batter", "season"], how="left")
        g_nb = nb["g"].to_numpy().astype(np.int64)
        num_nb = np.bincount(g_nb, weights=nb["woba_value"].to_numpy(), minlength=G)
        den_nb = np.bincount(g_nb, weights=nb["woba_denom"].to_numpy(), minlength=G)
    else:
        num_nb = np.zeros(G)
        den_nb = np.zeros(G)

    den = den_bbe + den_nb
    xw = (num + num_nb[None, :]) / np.clip(den, 1.0, None)[None, :]
    return keys.drop("g").with_columns(
        PA=pl.Series(den.astype(np.int64)),
        xwoba_mean=pl.Series(xw.mean(axis=0)),
        xwoba_sd=pl.Series(xw.std(axis=0, ddof=1) if S > 1 else np.zeros(G)),
        xwoba_q05=pl.Series(np.quantile(xw, 0.05, axis=0)),
        xwoba_q95=pl.Series(np.quantile(xw, 0.95, axis=0)),
    )


def build_player_table(rollup: pl.DataFrame, expected: pl.DataFrame) -> pl.DataFrame:
    """Join display names (KIT resolver; raw-id fallback per spec §8.4) and public xwOBA."""
    from pipeline.player_names import resolve_player_names

    ids = [int(i) for i in rollup["batter"].unique().to_list()]
    names = resolve_player_names(ids)   # omits unresolved ids; never raises
    return (
        rollup.with_columns(
            player_name=pl.col("batter").map_elements(
                lambda b: names.get(int(b), str(b)), return_dtype=pl.Utf8
            )
        )
        .join(
            expected.select("player_id", "season", "est_woba"),
            left_on=["batter", "season"], right_on=["player_id", "season"], how="left",
        )
        .rename({"est_woba": "xwoba_savant"})
        .select("batter", "player_name", "season", "PA",
                "xwoba_mean", "xwoba_sd", "xwoba_q05", "xwoba_q95", "xwoba_savant")
        .sort(["season", "xwoba_mean"], descending=[False, True])
    )
