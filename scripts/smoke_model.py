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
    logits[:, 4] = 0.10 * (X[:, 0] - 100) - 0.002 * (X[:, 1] - 28) ** 2
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
