"""Task A (v1 handoff): uncertainty sanity-check + model-vs-Savant disagreement leaderboard.

NO MCMC / NO re-fit. Reads the frozen v0 Stage C player table and answers two questions:

  1. Do the per-player posterior credible intervals behave? Regress
     log(interval width) on log(PA) for 100+ PA player-seasons; a ~1/sqrt(PA)
     shrinkage implies a slope near -0.5.
  2. Where does the model most disagree with public Savant xwOBA? Rank 100+ PA
     player-seasons by |xwoba_mean - xwoba_savant|, enrich each with sprint speed
     and batted-ball mix (from the local caches), and eyeball a few by hand.

Deliverables: results/task_a/{figures/*.png, task_a_metrics.json, disagreement_top.csv, NOTES.md}.
Run from repo root: `.venv/bin/python scripts/task_a_uncertainty.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# --- palette (colorblind-safe; matches the repo's plain-matplotlib figure style) ---
C_POINT = "#4878CF"   # scatter points (single series)
C_FIT = "#EE854A"     # fitted line
C_REF = "#8a8a8a"     # reference / grid
C_POS = "#b2182b"     # model HIGHER than Savant (diverging warm pole)
C_NEG = "#2166ac"     # model LOWER than Savant (diverging cool pole)

MIN_PA = 100
TOP_N = 25


# ---------------------------------------------------------------------------
# Pure helpers (unit-checkable; exercised by _selftest below)
# ---------------------------------------------------------------------------
def add_width(df: pl.DataFrame) -> pl.DataFrame:
    """Interval width = q95 - q05, plus signed disagreement vs Savant."""
    return df.with_columns(
        width=pl.col("xwoba_q95") - pl.col("xwoba_q05"),
        diff=pl.col("xwoba_mean") - pl.col("xwoba_savant"),
    )


def fit_loglog_slope(pa: np.ndarray, width: np.ndarray, seed: int = 42,
                     n_boot: int = 2000) -> dict:
    """OLS of log(width) on log(PA); returns slope, intercept, R^2 and a
    bootstrap 95% CI for the slope. Expect slope ~ -0.5 under 1/sqrt(PA)."""
    m = np.isfinite(pa) & np.isfinite(width) & (pa > 0) & (width > 0)
    x = np.log(pa[m].astype(float))
    y = np.log(width[m].astype(float))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = np.polyfit(x[idx], y[idx], 1)[0]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n": int(n),
        "slope": float(slope),
        "slope_ci95": [float(lo), float(hi)],
        "intercept": float(intercept),
        "r2": float(r2),
    }


def coverage_fraction(df: pl.DataFrame) -> float:
    """Fraction of player-seasons whose Savant xwOBA falls inside the model's
    90% credible interval [q05, q95]. NOTE: the interval is the posterior interval
    for the MODEL's own xwOBA (BART uncertainty), not a predictive interval for
    Savant's metric, so this measures model<->Savant agreement, not calibration."""
    inside = df.select(
        ((pl.col("xwoba_savant") >= pl.col("xwoba_q05"))
         & (pl.col("xwoba_savant") <= pl.col("xwoba_q95"))).mean()
    ).item()
    return float(inside)


# ---------------------------------------------------------------------------
# Enrichment from local caches (best-effort; skipped if caches absent)
# ---------------------------------------------------------------------------
def load_sprint(raw_dir: Path, seasons: list[int]) -> pl.DataFrame | None:
    frames = []
    for y in seasons:
        p = raw_dir / f"sprint_speed-{y}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return None
    return pl.concat(frames).select("player_id", "season", "sprint_speed")


def bbe_mix(raw_dir: Path, seasons: list[int], batters: set[int]) -> pl.DataFrame | None:
    """Per (batter, season): n_bbe, mean launch_speed, groundball rate — for the
    given batters only, straight from the slim Statcast caches."""
    from src.prep import filter_bbe

    frames = []
    for y in seasons:
        p = raw_dir / f"statcast-{y}-slim.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p).filter(pl.col("batter").is_in(list(batters)))
        if df.height == 0:
            continue
        bbe, _ = filter_bbe(df)
        frames.append(
            bbe.group_by("batter", "game_year").agg(
                n_bbe=pl.len(),
                mean_ev=pl.col("launch_speed").mean(),
                gb_rate=(pl.col("bb_type") == "ground_ball").mean(),
            ).rename({"game_year": "season"})
        )
    if not frames:
        return None
    return pl.concat(frames)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def pa_bin_medians(df: pl.DataFrame) -> list[dict]:
    """Median CI width within PA bins — an assumption-light view of the trend."""
    edges = [100, 150, 200, 300, 400, 500, 750]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sub = df.filter((pl.col("PA") >= a) & (pl.col("PA") < b))
        if sub.height:
            out.append({"pa_lo": a, "pa_hi": b, "n": sub.height,
                        "pa_mid": float(sub["PA"].median()),
                        "median_width": float(sub["width"].median())})
    return out


def fig_width_vs_pa(figdir: Path, df: pl.DataFrame, fit: dict, bins: list[dict]) -> None:
    pa = df["PA"].to_numpy().astype(float)
    width = df["width"].to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pa, width, s=10, alpha=0.22, color=C_POINT, edgecolors="none",
               label=f"player-seasons ({len(df):,})")
    # PA-bin medians make the (non-)trend unmistakable
    ax.plot([b["pa_mid"] for b in bins], [b["median_width"] for b in bins],
            "o-", color="#111111", lw=2, ms=7, label="median width per PA bin")
    xs = np.array([pa.min(), pa.max()])
    ax.plot(xs, np.exp(fit["intercept"]) * xs ** fit["slope"], color=C_FIT, lw=2.2,
            label=f"OLS fit: slope {fit['slope']:+.3f}  (95% CI {fit['slope_ci95'][0]:+.3f}, {fit['slope_ci95'][1]:+.3f})")
    # what a 1/sqrt(PA) sampling interval WOULD look like, anchored at the low-PA median
    anchor = bins[0]
    ax.plot(xs, anchor["median_width"] * (xs / anchor["pa_mid"]) ** -0.5,
            color=C_REF, ls="--", lw=1.5, label="expected if 1/√PA (slope −0.5)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("PA (log scale)")
    ax.set_ylabel("90% credible-interval width  (q95 − q05, log scale)")
    ax.set_title("v0 per-player interval width is ~flat in PA (not 1/√PA)")
    ax.grid(True, which="both", color=C_REF, alpha=0.15)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "interval_width_vs_pa.png", dpi=120)
    plt.close(fig)


def fig_disagreement(figdir: Path, top: pl.DataFrame) -> None:
    t = top.sort("diff")  # most negative at bottom -> ascending so largest bars read outward
    labels = [f"{n}  '{str(s)[2:]}  ({pa} PA)"
              for n, s, pa in zip(t["player_name"], t["season"], t["PA"])]
    diffs = t["diff"].to_numpy()
    colors = [C_POS if d > 0 else C_NEG for d in diffs]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(t) + 1.2))
    ax.barh(range(len(t)), diffs, color=colors, height=0.72)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("model xwOBA − Savant xwOBA")
    ax.set_title(f"Where v0 most disagrees with Savant  (top {len(t)}, {MIN_PA}+ PA)")
    ax.grid(True, axis="x", color=C_REF, alpha=0.2)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_POS, label="model higher"),
                       Patch(color=C_NEG, label="model lower")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "disagreement_leaderboard.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Synthetic check: width = 0.4 * PA^-0.5 * lognormal noise -> slope ~ -0.5."""
    rng = np.random.default_rng(0)
    pa = rng.integers(100, 700, 4000).astype(float)
    width = 0.4 * pa ** -0.5 * np.exp(rng.normal(0, 0.15, len(pa)))
    fit = fit_loglog_slope(pa, width, n_boot=200)
    assert abs(fit["slope"] + 0.5) < 0.03, fit
    assert fit["slope_ci95"][0] < -0.5 < fit["slope_ci95"][1], fit
    # add_width arithmetic
    d = add_width(pl.DataFrame({
        "xwoba_q95": [0.4], "xwoba_q05": [0.3],
        "xwoba_mean": [0.35], "xwoba_savant": [0.30],
    }))
    assert abs(d["width"][0] - 0.10) < 1e-12 and abs(d["diff"][0] - 0.05) < 1e-12
    print("  _selftest OK")


def main() -> None:
    from src.config import load_config

    _selftest()
    cfg = load_config()
    seasons = cfg.all_seasons

    pt = pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet")
    df = add_width(pt).filter((pl.col("PA") >= MIN_PA) & (pl.col("width") > 0))
    print(f"  {df.height:,} player-seasons with {MIN_PA}+ PA")

    # --- Part 1: interval width vs PA ---
    fit = fit_loglog_slope(df["PA"].to_numpy(), df["width"].to_numpy())
    bins = pa_bin_medians(df)
    cov = coverage_fraction(df)
    r_all = float(np.corrcoef(df["xwoba_mean"], df["xwoba_savant"])[0, 1])
    # width tracks the player's contact profile more than sample size:
    width_r_pa = float(np.corrcoef(df["width"], df["PA"].cast(pl.Float64))[0, 1])
    width_r_val = float(np.corrcoef(df["width"], df["xwoba_mean"])[0, 1])

    # --- tail compression: model shrinks the extremes toward the mean ---
    sv = df["xwoba_savant"].to_numpy()
    shrink_slope = float(np.polyfit(sv, df["xwoba_mean"].to_numpy(), 1)[0])
    diff_slope, diff_int = (float(v) for v in np.polyfit(sv, df["diff"].to_numpy(), 1))

    # --- Part 2: disagreement leaderboard ---
    ranked = df.with_columns(abs_diff=pl.col("diff").abs()).sort("abs_diff", descending=True)
    top = ranked.head(TOP_N)

    sprint = load_sprint(cfg.raw_dir, seasons)
    if sprint is not None:
        top = top.join(sprint, left_on=["batter", "season"],
                       right_on=["player_id", "season"], how="left")
    mix = bbe_mix(cfg.raw_dir, seasons, set(top["batter"].to_list()))
    if mix is not None:
        top = top.join(mix, on=["batter", "season"], how="left")

    disp_cols = ["player_name", "season", "PA", "xwoba_mean", "xwoba_savant", "diff",
                 "xwoba_q05", "xwoba_q95"]
    for opt in ("sprint_speed", "mean_ev", "gb_rate"):
        if opt in top.columns:
            disp_cols.append(opt)
    top_disp = top.select(disp_cols).sort("diff", descending=True)

    # --- write outputs ---
    outdir = cfg.results_dir / "task_a"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_width_vs_pa(figdir, df, fit, bins)
    fig_disagreement(figdir, top)
    top_disp.write_csv(outdir / "disagreement_top.csv")

    metrics = {
        "min_pa": MIN_PA,
        "n_player_seasons": df.height,
        "median_pa": float(df["PA"].median()),
        "median_width": float(df["width"].median()),
        "width_vs_pa_loglog": fit,
        "width_bin_medians": bins,
        "width_r_pa": width_r_pa,
        "width_r_xwoba_mean": width_r_val,
        "savant_in_ci90_fraction": cov,
        "player_r_all_seasons": r_all,
        "tail_compression": {
            "model_vs_savant_slope": shrink_slope,
            "diff_vs_savant_slope": diff_slope,
            "diff_vs_savant_intercept": diff_int,
        },
        "diff_mean_bias": float(df["diff"].mean()),
        "diff_mean_abs": float(df["diff"].abs().mean()),
        "diff_p95_abs": float(df["diff"].abs().quantile(0.95)),
    }
    (outdir / "task_a_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print("\n  top disagreements:")
    with pl.Config(tbl_rows=TOP_N, tbl_cols=-1, fmt_str_lengths=30):
        print(top_disp)
    print(f"\n  wrote {outdir}/ (figures/, task_a_metrics.json, disagreement_top.csv)")


if __name__ == "__main__":
    main()
