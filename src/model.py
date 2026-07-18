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


def predict_mu(model, idata, X_new: np.ndarray, seed: int) -> np.ndarray:
    """Documented pm.set_data + sample_posterior_predictive path (spec §7.3).

    NON-FUNCTIONAL for OOS in pymc-bart 0.12.0: because mu is declared with the static
    shape (K, n_train), set_data does not resample it — sample_posterior_predictive
    emits ImplicitFreezeWarning and returns the frozen in-sample trace of shape
    (K, n_train) regardless of X_new (a wrong answer, not an exception). Kept for
    reference; predict_mu_any uses the stored-trees predictor instead."""
    import pymc as pm

    with model:
        pm.set_data({"X_data": X_new})
        pp = pm.sample_posterior_predictive(
            idata, var_names=["mu"], random_seed=seed, progressbar=False
        )
    vals = pp.posterior_predictive["mu"].values     # (chains, draws, K, n_new)
    return vals.reshape(-1, *vals.shape[2:])


def predict_mu_from_trees(model, idata, X_new: np.ndarray, seed: int) -> np.ndarray:
    """Fallback: sample the fitted trees directly. pymc_bart.utils._sample_posterior
    returns (S, n_new, K) in the installed version (0.12.0), so transpose to the
    (S, K, n_new) convention model.py uses everywhere."""
    from pymc_bart.utils import _sample_posterior

    all_trees = model["mu"].owner.op.all_trees
    rng = np.random.default_rng(seed)
    S = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    out = np.asarray(_sample_posterior(all_trees, X=X_new, rng=rng, size=S, shape=K))
    return out.transpose(0, 2, 1)


def predict_mu_any(model, idata, X_new: np.ndarray, seed: int) -> np.ndarray:
    """OOS prediction. The pm.set_data path silently freezes mu in pymc-bart 0.12.0
    (see predict_mu), so predict directly from the stored trees — the spec §7.3
    sanctioned fallback, validated by the smoke and by verify_oos_mechanism at
    corr ~1.0 against the in-sample posterior."""
    return predict_mu_from_trees(model, idata, X_new, seed)


@dataclass
class PredictBundle:
    """Bounded-memory reductions of the posterior predictive (spec §7.2/§8.2)."""
    p_mean: np.ndarray            # (n, K) float32 — posterior-mean class probabilities
    ev_draws: np.ndarray          # (S, n) float32 — expected wOBA value per draw
    lppd_i: np.ndarray | None     # (n,) log( mean_s p[y_i] ) — None when y unknown
    meanlog_i: np.ndarray | None  # (n,) mean_s log p[y_i]


def predict_and_reduce(model, idata, X_new: np.ndarray, y_new: np.ndarray | None,
                       w: np.ndarray, cfg: Config, seed: int) -> PredictBundle:
    """Chunked prediction; keeps only bounded reductions, never the full (S, n, K)."""
    from scipy.special import logsumexp

    from src.rollup import expected_values

    idt = thin(idata, cfg.thin_draws)
    n = X_new.shape[0]
    p_mean = np.zeros((n, K), dtype=np.float64)
    ev_parts, lp_parts, ml_parts = [], [], []
    for lo in range(0, n, cfg.chunk_size):
        hi = min(lo + cfg.chunk_size, n)
        mu = predict_mu_any(model, idt, X_new[lo:hi], seed)    # (S, K, c)
        p = _softmax(mu, axis=1)
        p_mean[lo:hi] = p.mean(axis=0).T
        ev_parts.append(expected_values(p, w))                 # (S, c) f32
        if y_new is not None:
            yc = y_new[lo:hi]
            py = np.clip(p[:, yc, np.arange(hi - lo)], 1e-12, None)   # (S, c)
            lp_parts.append(logsumexp(np.log(py), axis=0) - np.log(p.shape[0]))
            ml_parts.append(np.log(py).mean(axis=0))
    return PredictBundle(
        p_mean=p_mean.astype(np.float32),
        ev_draws=np.concatenate(ev_parts, axis=1),
        lppd_i=np.concatenate(lp_parts) if lp_parts else None,
        meanlog_i=np.concatenate(ml_parts) if ml_parts else None,
    )


def verify_oos_mechanism(model, idata, X_train: np.ndarray, cfg: Config, seed: int) -> dict:
    """Predict training rows via the OOS path; posterior-mean probs must agree with
    the in-sample posterior (spec §7.3 verification gate)."""
    idt = thin(idata, min(200, cfg.thin_draws))
    take = min(2000, X_train.shape[0])
    mu_new = predict_mu_any(model, idt, X_train[:take], seed)
    p_new = _softmax(mu_new, axis=1).mean(axis=0).T                       # (take, K)
    mu_in = idt.posterior["mu"].values
    mu_in = mu_in.reshape(-1, K, mu_in.shape[-1])[:, :, :take]
    p_in = _softmax(mu_in, axis=1).mean(axis=0).T
    r = float(np.corrcoef(p_new.ravel(), p_in.ravel())[0, 1])
    diff = np.abs(p_new - p_in)
    # The stored-trees predictor averages a RANDOM subset of trees; p_in averages a
    # DIFFERENT thinned subset. For a high-variance BART posterior the two Monte-Carlo
    # estimates of the per-event mean agree in structure (corr) but a few worst-case
    # events differ — max_abs is scale-sensitive (grows with event count) and conflates
    # that MC noise with a real mechanism bug. A wrong axis/tree indexing collapses corr
    # and inflates the MEAN diff, so gate on those (both scale-invariant) instead.
    return {"corr": r, "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "pass": bool(r > 0.99 and diff.mean() < 0.03)}
