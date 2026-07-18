"""BART categorical model: build/fit, post-fit sanity, thinning, OOS prediction."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Config, StageConfig
from src.prep import K


def _softmax(z: np.ndarray, axis: int) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit(X: np.ndarray, y: np.ndarray, stage: StageConfig, seed: int):
    """Fit the pymc-bart multiclass model (spec §7). Returns (model, idata, runtime_s)."""
    import pymc as pm
    import pymc_bart as pmb

    n = X.shape[0]
    t0 = time.perf_counter()
    with pm.Model() as model:
        X_data = pm.Data("X_data", X)
        mu = pmb.BART("mu", X_data, y, m=stage.m_trees, shape=(K, n))
        if stage.store_p:  # Stage A wiring proof only — doubles idata size (spec §7.2)
            p = pm.Deterministic("p", pm.math.softmax(mu, axis=0))
            p_t = p.T
        else:
            p_t = pm.math.softmax(mu, axis=0).T
        pm.Categorical("y_obs", p=p_t, observed=y)
        idata = pm.sample(
            tune=stage.tune, draws=stage.draws, chains=stage.chains,
            random_seed=seed, compute_convergence_checks=True,
        )
    return model, idata, time.perf_counter() - t0


def sanity_check(idata, n_probe: int = 2000, seed: int = 0) -> list[str]:
    """Cheap post-fit diagnostics. Full R-hat over every mu cell is impractical at
    scale, so probe a random subset; also detect collapsed class probabilities.
    Non-empty return = stop and report (spec §7.4)."""
    import arviz as az

    warnings: list[str] = []
    mu = idata.posterior["mu"]                      # dims: (chain, draw, K, n)
    vals = mu.values
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(vals.shape[-1], size=min(n_probe, vals.shape[-1]), replace=False))
    r = az.rhat(vals[:, :, :, idx])
    rhat_max = float(r.to_array().max()) if hasattr(r, "to_array") else float(np.max(np.asarray(r)))
    if rhat_max > 1.1:
        warnings.append(f"max R-hat on probed mu cells = {rhat_max:.3f} (> 1.1)")
    p_mean = _softmax(vals.mean(axis=(0, 1)), axis=0)   # (K, n)
    class_means = p_mean.mean(axis=1)
    if (class_means < 1e-4).any():
        warnings.append(f"collapsed class probabilities: {np.round(class_means, 5).tolist()}")
    return warnings


def thin(idata, max_total_draws: int):
    """Thin posterior draws to at most max_total_draws across chains."""
    chains = idata.posterior.sizes["chain"]
    draws = idata.posterior.sizes["draw"]
    per_chain = max(1, max_total_draws // chains)
    if draws <= per_chain:
        return idata
    step = int(np.ceil(draws / per_chain))
    return idata.isel(draw=slice(None, None, step))


def save_idata(idata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(path))
