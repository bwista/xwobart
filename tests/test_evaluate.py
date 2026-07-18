import numpy as np

from src.evaluate import brier_score, elpd_metrics, reliability_curve


def test_reliability_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.uniform(size=20_000) < p).astype(float)
    curve = reliability_curve(p, y, n_bins=10)
    assert curve["ece"] < 0.02
    assert len(curve["conf"]) <= 10 and len(curve["conf"]) == len(curve["acc"])
    assert sum(curve["count"]) == 20_000


def test_reliability_degenerate_probs_no_crash():
    # Triple-like: nearly all mass at ~0 -> duplicate quantile edges must collapse
    p = np.concatenate([np.full(9_990, 0.001), np.full(10, 0.4)])
    y = np.zeros(10_000)
    curve = reliability_curve(p, y, n_bins=10)
    assert np.isfinite(curve["ece"])
    assert sum(curve["count"]) == 10_000


def test_brier():
    p = np.array([1.0, 0.0, 0.5])
    y = np.array([1.0, 0.0, 1.0])
    assert abs(brier_score(p, y) - (0.25 / 3)) < 1e-9


def test_elpd_known_values():
    # Two events, two draws, K=2. Event 0: p[y]=0.5 both draws. Event 1: 0.25 / 0.75.
    lppd0 = np.log(0.5)
    lppd1 = np.log((0.25 + 0.75) / 2)
    lppd_i = np.array([lppd0, lppd1])
    meanlog_i = np.array([np.log(0.5), (np.log(0.25) + np.log(0.75)) / 2])
    m = elpd_metrics(lppd_i, meanlog_i)
    assert abs(m["elpd_lppd"] - (lppd0 + lppd1)) < 1e-12
    assert abs(m["elpd_se"] - np.sqrt(2 * np.var(lppd_i))) < 1e-12
    assert m["mean_log_lik_sum"] < m["elpd_lppd"]   # Jensen: mean-of-log <= log-of-mean
    assert m["n_events"] == 2
