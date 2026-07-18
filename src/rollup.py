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
