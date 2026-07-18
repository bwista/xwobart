"""Synthetic-data smoke: proves fit() + sanity_check() work in ~1-3 minutes.
Run: .venv/bin/python scripts/smoke_model.py

All executable code sits under `if __name__ == "__main__":` — pymc-bart's Manager and
pymc's parallel chains use the 'spawn' start method (macOS default), which re-imports
this module in child processes; the guard stops that re-import from re-running the fit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import StageConfig
from src.model import _softmax, fit, sanity_check

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800
    X = np.column_stack([
        rng.uniform(60, 115, n),    # launch_speed
        rng.uniform(-60, 60, n),    # launch_angle
        rng.uniform(23, 31, n),     # sprint_speed
    ])
    # Toy structure: HRs need hard+elevated contact; grounder singles need speed.
    logits = np.zeros((n, 5))
    logits[:, 0] = 1.0
    logits[:, 1] = 0.02 * (X[:, 0] - 85) + 0.15 * (X[:, 2] - 27) * (X[:, 1] < 0)
    logits[:, 2] = 0.03 * (X[:, 0] - 95)
    logits[:, 3] = -2.0
    # Strong, separable EV -> HR signal so the tiny (100-draw, 10-tree) fit learns it
    # robustly despite spawn's non-reproducible RNG (was 0.10*(EV-100): too marginal
    # with only ~26 HR examples, flipping the assertion below run-to-run).
    logits[:, 4] = 0.20 * (X[:, 0] - 95) - 0.002 * (X[:, 1] - 28) ** 2
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(5, p=p_true[i]) for i in range(n)])
    print("class counts:", np.bincount(y, minlength=5))

    stage = StageConfig(name="smoke", subsample=None, m_trees=10, tune=100, draws=100,
                        chains=2, store_p=True, predict_cap=None)
    model, idata, secs = fit(X, y, stage, seed=42)
    print(f"fit runtime: {secs:.1f}s")
    warnings = sanity_check(idata)
    print("sanity warnings:", warnings or "none")

    mu = idata.posterior["mu"].values          # (chains, draws, 5, n)
    p_mean = _softmax(mu.mean(axis=(0, 1)), axis=0)
    hr_hard = p_mean[4, X[:, 0] > 105].mean()
    hr_soft = p_mean[4, X[:, 0] < 75].mean()
    print(f"P(HR | EV>105) = {hr_hard:.3f}  vs  P(HR | EV<75) = {hr_soft:.3f}")
    assert hr_hard > hr_soft + 0.05, "model failed to learn the obvious EV -> HR signal"
    assert p_mean.shape == (5, n)
    print("SMOKE OK")

    # ---- Task 12: OOS prediction ----
    from src.config import Config  # noqa: E402
    from src.model import (  # noqa: E402
        _softmax as _sm,
        predict_and_reduce,
        predict_mu_from_trees,
        verify_oos_mechanism,
    )

    cfg_like = Config(
        seed=42, train_seasons=[2022], holdout_season=2025,
        statcast_dir=Path("."), raw_dir=Path("."), results_dir=Path("."),
        season_windows={}, sprint_min_opp=10, stages={},
        thin_draws=100, chunk_size=300, min_pa=100, reliability_bins=10,
        sprint_grid=(23.0, 31.0, 17),
    )
    check = verify_oos_mechanism(model, idata, X, cfg_like, seed=42)
    print("OOS verification:", check)
    assert check["pass"], "OOS prediction disagrees with in-sample posterior"

    # Decisively exercise the stored-trees FALLBACK (n_new != n_train): its posterior-mean
    # probs on a slice must correlate with the in-sample posterior for those rows. This
    # catches an axis-order mistake in the (S, n, K) -> (S, K, n) transpose that the weaker
    # predict_and_reduce asserts below would silently pass.
    mu_fb = predict_mu_from_trees(model, idata, X[:300], seed=42)   # (S, K, 300)
    p_fb = _sm(mu_fb, axis=1).mean(axis=0).T                        # (300, K)
    mu_in300 = idata.posterior["mu"].values.reshape(-1, 5, X.shape[0])[:, :, :300]
    p_in300 = _sm(mu_in300, axis=1).mean(axis=0).T
    fb_corr = float(np.corrcoef(p_fb.ravel(), p_in300.ravel())[0, 1])
    print(f"stored-trees fallback corr vs in-sample (n=300): {fb_corr:.4f}")
    assert fb_corr > 0.99, f"fallback axis order likely wrong (corr={fb_corr:.4f})"

    w = np.array([0.0, 0.9, 1.25, 1.6, 2.0])
    bundle = predict_and_reduce(model, idata, X[:500], y[:500], w, cfg_like, seed=42)
    assert bundle.p_mean.shape == (500, 5)
    assert bundle.ev_draws.shape[1] == 500
    assert np.all(bundle.ev_draws >= 0) and np.all(bundle.ev_draws <= 2.0)
    assert bundle.lppd_i.shape == (500,) and np.all(bundle.lppd_i <= 0)
    print("PREDICT OK")
