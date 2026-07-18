"""The four v0 acceptance checks (spec §9). Pure metric helpers are unit-tested;
figure functions are exercised by the Stage A wiring run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.prep import CLASS_NAMES, K


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def reliability_curve(p: np.ndarray, y_ind: np.ndarray, n_bins: int) -> dict:
    """Quantile-binned reliability. Duplicate quantile edges (degenerate classes like
    triples) collapse gracefully; empty bins are skipped (spec §9.2)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    conf, acc, count = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        conf.append(float(p[mask].mean()))
        acc.append(float(y_ind[mask].mean()))
        count.append(int(mask.sum()))
    n = len(p)
    ece = float(sum(c / n * abs(a - cf) for cf, a, c in zip(conf, acc, count)))
    return {"conf": conf, "acc": acc, "count": count, "ece": ece}


def brier_score(p: np.ndarray, y_ind: np.ndarray) -> float:
    return float(np.mean((p - y_ind) ** 2))


def calibration(figdir: Path, p_mean: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
    """Per-class reliability curves + Brier + ECE on the holdout; weighted aggregate."""
    out: dict = {"per_class": {}}
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), sharey=True)
    freq = np.bincount(y, minlength=K) / len(y)
    for c in range(K):
        y_ind = (y == c).astype(float)
        curve = reliability_curve(p_mean[:, c], y_ind, n_bins)
        out["per_class"][CLASS_NAMES[c]] = {
            "brier": brier_score(p_mean[:, c], y_ind),
            "ece": curve["ece"],
            "bin_counts": curve["count"],
        }
        ax = axes[c]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(curve["conf"], curve["acc"], "o-")
        ax.set_title(f"{CLASS_NAMES[c]} (n bins={len(curve['count'])})")
        ax.set_xlabel("predicted prob")
        lim = max(max(curve["conf"], default=0.1), max(curve["acc"], default=0.1)) * 1.15
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    axes[0].set_ylabel("observed frequency")
    fig.tight_layout()
    fig.savefig(figdir / "calibration_reliability.png", dpi=120)
    plt.close(fig)
    out["ece_weighted"] = float(sum(freq[c] * out["per_class"][CLASS_NAMES[c]]["ece"] for c in range(K)))
    return out


def elpd_metrics(lppd_i: np.ndarray, meanlog_i: np.ndarray) -> dict:
    """Primary anchor: lppd (log-of-mean) + SE. Mean-of-log stored too (spec §9.4)."""
    n = len(lppd_i)
    return {
        "elpd_lppd": float(lppd_i.sum()),
        "elpd_se": float(np.sqrt(n * np.var(lppd_i))),
        "mean_log_lik_sum": float(meanlog_i.sum()),
        "n_events": int(n),
    }
