"""Task 3 de-risk pre-check (spec: docs/superpowers/plans/2026-07-23-forecast-
rungb-spray.md, §5): does a hitter's early-season pull tendency predict his
rest-of-season xwOBA BEYOND what his early xwOBA + exit-velocity + barrel-rate
already say? Pure closed-form OLS, no BART -- this question has to be answered
before any model gets built. Orchestration (real-data assembly, GO/STOP verdict)
lives in scripts/run_talent3.py's --precheck branch."""
from __future__ import annotations

import numpy as np

BASE_KEYS = ["xwoba", "ev", "barrel"]   # design order after the intercept


def pull_incremental_signal(X: dict, y: np.ndarray) -> dict:
    """Two OLS fits on the same eligible rows: base = [1, xwoba, ev, barrel],
    full = base + pull. Rows with any NaN feature (X or y) are dropped first --
    a player with zero tracked BBE in his early-k PAs has null ev/barrel/pull.

    r2 = 1 - SSR/SST (SST relative to the fitted sample's own mean, so r2_base
    and r2_full are on a common, comparable footing). delta_r2 = r2_full -
    r2_base is the headline number: incremental variance explained by pull once
    xwoba/ev/barrel already get their say.

    pull_partial_corr is the partial correlation of pull with y controlling for
    the base regressors, via Frisch-Waugh-Lovell: residualize both pull and y on
    the base design, then correlate the residuals. Chosen (over the pull
    coefficient's t-stat) because it is scale-free and interpretable on its own
    (e.g. "pull correlates 0.45 with rest-of-season xwOBA after netting out
    early xwOBA/ev/barrel"). Internal identity for a single added regressor:
    delta_r2 == pull_partial_corr**2 * (1 - r2_base) -- pull_partial_corr**2 is
    the share of the BASE MODEL'S RESIDUAL variance that pull explains, whereas
    delta_r2 is that same gain expressed as a share of TOTAL variance; verified
    numerically, do not conflate the two.

    Returns {r2_base, r2_full, delta_r2, pull_partial_corr, n}."""
    xwoba = np.asarray(X["xwoba"], dtype=float)
    ev = np.asarray(X["ev"], dtype=float)
    barrel = np.asarray(X["barrel"], dtype=float)
    pull = np.asarray(X["pull"], dtype=float)
    y = np.asarray(y, dtype=float)

    stacked = np.column_stack([xwoba, ev, barrel, pull, y])
    keep = ~np.isnan(stacked).any(axis=1)
    xwoba, ev, barrel, pull, y = (stacked[keep, j] for j in range(5))
    n = int(keep.sum())

    ones = np.ones(n)
    base = np.column_stack([ones, xwoba, ev, barrel])
    full = np.column_stack([base, pull])
    sst = float(np.sum((y - y.mean()) ** 2))

    def r2_of(design: np.ndarray) -> tuple[float, np.ndarray]:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        ssr = float(np.sum(resid ** 2))
        return (1.0 - ssr / sst if sst > 0 else 0.0), coef

    r2_base, coef_base = r2_of(base)
    r2_full, _ = r2_of(full)

    # partial correlation of pull with y | base (Frisch-Waugh-Lovell)
    pull_coef, *_ = np.linalg.lstsq(base, pull, rcond=None)
    pull_resid = pull - base @ pull_coef
    y_resid = y - base @ coef_base
    denom = np.linalg.norm(pull_resid) * np.linalg.norm(y_resid)
    pull_partial_corr = float(pull_resid @ y_resid / denom) if denom > 0 else 0.0

    return {
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "delta_r2": float(r2_full - r2_base),
        "pull_partial_corr": pull_partial_corr,
        "n": n,
    }
