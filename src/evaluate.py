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


def binned_residuals(model_val: np.ndarray, public_val: np.ndarray, x: np.ndarray,
                     lo: float, hi: float, width: float) -> list[dict]:
    """Mean (model - public) residual per x-bin — exposes structure a single r hides."""
    out = []
    edges = np.arange(lo, hi + width, width)
    res = model_val - public_val
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(res)
        if m.sum() >= 25:
            out.append({"bin_lo": float(a), "bin_hi": float(b),
                        "mean_residual": float(res[m].mean()), "n": int(m.sum())})
    return out


def replication(figdir: Path, *, ev_mean_train, public_train, ev_mean_hold, public_hold,
                ls_hold, la_hold, player_table: pl.DataFrame, min_pa: int,
                train_seasons: list[int], seed: int) -> dict:
    out = {
        "event_r_train": pearson(ev_mean_train, public_train),
        "event_r_holdout": pearson(ev_mean_hold, public_hold),
    }
    rng = np.random.default_rng(seed)
    for tag, mv, pv in (("train", ev_mean_train, public_train),
                        ("holdout", ev_mean_hold, public_hold)):
        m = np.isfinite(mv) & np.isfinite(pv)
        idx = np.flatnonzero(m)
        idx = rng.choice(idx, size=min(20_000, len(idx)), replace=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pv[idx], mv[idx], s=3, alpha=0.15)
        ax.plot([0, 2.1], [0, 2.1], "--", color="grey", lw=1)
        ax.set_xlabel("estimated_woba_using_speedangle (Savant)")
        ax.set_ylabel("posterior-mean expected value (model)")
        ax.set_title(f"event-level replication — {tag} (r={out[f'event_r_{tag}']:.3f})")
        fig.savefig(figdir / f"replication_event_{tag}.png", dpi=120)
        plt.close(fig)

    pt = player_table.filter(pl.col("PA") >= min_pa).drop_nulls("xwoba_savant")
    tr = pt.filter(pl.col("season").is_in(train_seasons))
    ho = pt.filter(~pl.col("season").is_in(train_seasons))
    out["player_r_train"] = pearson(tr["xwoba_mean"].to_numpy(), tr["xwoba_savant"].to_numpy())
    out["player_r_holdout"] = pearson(ho["xwoba_mean"].to_numpy(), ho["xwoba_savant"].to_numpy())
    out["players_train"], out["players_holdout"] = tr.height, ho.height
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tr["xwoba_savant"], tr["xwoba_mean"], s=8, alpha=0.4, label=f"train (r={out['player_r_train']:.3f})")
    ax.scatter(ho["xwoba_savant"], ho["xwoba_mean"], s=8, alpha=0.4, label=f"holdout (r={out['player_r_holdout']:.3f})")
    ax.plot([0.2, 0.5], [0.2, 0.5], "--", color="grey", lw=1)
    ax.set_xlabel("public xwOBA (Savant)"); ax.set_ylabel("rollup posterior mean")
    ax.legend(); ax.set_title(f"player-season replication ({min_pa}+ PA)")
    fig.savefig(figdir / "replication_player.png", dpi=120)
    plt.close(fig)

    out["residual_by_ev"] = binned_residuals(ev_mean_hold, public_hold, ls_hold, 40, 120, 2.5)
    out["residual_by_la"] = binned_residuals(ev_mean_hold, public_hold, la_hold, -75, 75, 5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, lab in ((axes[0], "residual_by_ev", "launch_speed (mph)"),
                         (axes[1], "residual_by_la", "launch_angle (deg)")):
        rows = out[key]
        ax.axhline(0, color="grey", lw=1)
        ax.plot([r["bin_lo"] for r in rows], [r["mean_residual"] for r in rows], "o-")
        ax.set_xlabel(lab); ax.set_ylabel("mean(model - public)")
    fig.suptitle("holdout residual structure vs EV / LA bins")
    fig.savefig(figdir / "replication_residual_bins.png", dpi=120)
    plt.close(fig)
    return out


def undercorrection(actual: np.ndarray, model_pred: np.ndarray, public_pred: np.ndarray,
                    sprint: np.ndarray) -> dict:
    """Spec §9.4: on ground balls, residual-vs-sprint correlation, model vs public.
    The public metric undercorrects for speed; the model should shrink toward zero."""
    return {
        "model_residual_sprint_corr": pearson(actual - model_pred, sprint),
        "public_residual_sprint_corr": pearson(actual - public_pred, sprint),
    }


SPRAY_PULL_DEG = 20.0     # ~ the HR-region pull angle measured on 2022-25


def contact_grids(grid_cfg: tuple[float, float, int], variant: str = "v0"):
    """Fixed contact points crossed with a sprint-speed grid (spec §9.3):
    topped grounder (85, -10) and barrel (103, 28).

    variant 'v0'    -> (s, grounder(n,3), barrel(n,3)) -- unchanged, 3 features.
    variant 'spray' -> (s, {name: (n,5)}) with the SAME grounder hit PULLED (+20 deg)
    and OPPOSITE (-20 deg). All grids are RHB (stand_R = 1.0). The pull/oppo pair is
    the E7 diagnostic: once the surface can see direction, sprint speed's payoff should
    concentrate on pulled grounders (the ones a fast runner beats out)."""
    lo, hi, n = grid_cfg
    n = int(n)
    s = np.linspace(lo, hi, n)
    if variant == "v0":
        grounder = np.column_stack([np.full(n, 85.0), np.full(n, -10.0), s])
        barrel = np.column_stack([np.full(n, 103.0), np.full(n, 28.0), s])
        return s, grounder, barrel

    def g(ls: float, la: float, spray: float) -> np.ndarray:
        return np.column_stack([np.full(n, ls), np.full(n, la), np.full(n, spray),
                                np.ones(n), s])

    return s, {"grounder_pull": g(85.0, -10.0, SPRAY_PULL_DEG),
               "grounder_oppo": g(85.0, -10.0, -SPRAY_PULL_DEG),
               "barrel_pull": g(103.0, 28.0, SPRAY_PULL_DEG)}


def la_spray_grid(la: tuple[float, float, int] = (0.0, 45.0, 19),
                  spray: tuple[float, float, int] = (-45.0, 45.0, 37),
                  launch_speed: float = 103.0, sprint: float = 27.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LA x spray grid at fixed EV for the spec's HR-band partial-dependence plot
    (spec §"Recommended spec": "PDP/importance for spray (HR band in LA x spray)").
    Returns (la_axis, spray_axis, X (n_la*n_spray, 5)) in row-major order, RHB."""
    la_ax = np.linspace(*la[:2], int(la[2]))
    sp_ax = np.linspace(*spray[:2], int(spray[2]))
    L, S = np.meshgrid(la_ax, sp_ax, indexing="ij")
    n = L.size
    X = np.column_stack([np.full(n, launch_speed), L.ravel(), S.ravel(),
                         np.ones(n), np.full(n, sprint)])
    return la_ax, sp_ax, X


def localization(figdir: Path, s: np.ndarray, grounder_ev: np.ndarray, barrel_ev: np.ndarray,
                 sprint_hold: np.ndarray, codes_hold: np.ndarray, ev_mean_hold: np.ndarray) -> dict:
    """grounder_ev / barrel_ev are (S, len(s)) per-draw expected values on the grids."""
    out: dict = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, evd, color in (("topped grounder (85, -10)", grounder_ev, "C0"),
                             ("barrel (103, 28)", barrel_ev, "C1")):
        mean = evd.mean(axis=0)
        q05, q95 = np.quantile(evd, [0.05, 0.95], axis=0)
        ax.plot(s, mean, color=color, label=name)
        ax.fill_between(s, q05, q95, color=color, alpha=0.2)
        slope = float(np.polyfit(s, mean, 1)[0])
        out[("grounder" if "grounder" in name else "barrel") + "_slope_per_ftps"] = slope
    ax.set_xlabel("sprint speed (ft/s)"); ax.set_ylabel("expected wOBA value")
    ax.legend(); ax.set_title("sprint-speed localization (90% credible bands)")
    fig.savefig(figdir / "sprint_localization_curves.png", dpi=120)
    plt.close(fig)

    for label, codes in (("weak_topped", (1, 2)), ("solid_barrel", (5, 6))):
        m = np.isin(codes_hold, codes) & np.isfinite(sprint_hold) & np.isfinite(ev_mean_hold)
        out[f"{label}_sprint_corr"] = pearson(ev_mean_hold[m], sprint_hold[m])
        out[f"{label}_sprint_slope"] = float(np.polyfit(sprint_hold[m], ev_mean_hold[m], 1)[0])
        out[f"{label}_n"] = int(m.sum())
    return out


def variable_importance(figdir: Path, model, idata, X: np.ndarray,
                        labels: list[str] | None = None) -> dict:
    """pymc-bart variable importance for the three features. The API has moved between
    versions — adapt the call to the installed version if it raises; metrics tolerate
    an 'unavailable' record but Stage B should ship real numbers (spec §9.3)."""
    try:
        import pymc_bart as pmb

        vi = pmb.compute_variable_importance(idata, model["mu"], X)
        axes = pmb.plot_variable_importance(vi)
        fig = (axes[0] if isinstance(axes, (list, np.ndarray)) else axes).figure
        fig.savefig(figdir / "variable_importance.png", dpi=120)
        plt.close(fig)
        labels = labels or ["launch_speed", "launch_angle", "sprint_speed"]
        # Keep only small summary fields. compute_variable_importance also returns large
        # per-event `preds`/`preds_all` arrays (for plotting) that bloat metrics.json to
        # ~100s of MB — never serialize those.
        keep = {"indices", "labels", "r2_mean", "r2_hdi"}
        return {"method": "pymc_bart.compute_variable_importance",
                "feature_labels": labels,
                "raw": {k: np.asarray(v).tolist() for k, v in vi.items()
                        if k in keep and isinstance(v, (list, np.ndarray))}}
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
