"""Sample-size-aware per-player xwOBA credible interval — the band that answers
"given the N balls this hitter has put up, where could their true xwOBA be?"

v0's model interval does NOT do this: it tracks BART surface uncertainty at a
player's contact profile and is ~flat in PA (Task A). Here we build the *sampling*
band instead. Per player-season, every plate appearance gets an xwOBA value:

  * batted ball -> estimated_woba_using_speedangle (Savant per-BBE xwOBA;
                   falls back to woba_value if the BBE estimate is missing)
  * walk/K/HBP  -> woba_value (deterministic)

xwOBA = Σvalue / Σwoba_denom. We **bootstrap-resample the player's PAs** and recompute
xwOBA B times; the 5th/95th percentiles are the band. It narrows as ~1/√PA by
construction, is naturally asymmetric (a boom-or-bust hitter's band is wider), and
needs NO re-fit. Base values are Savant's (v0 is at parity with Savant, and the
model's per-event EVs are not persisted) — the band *width* is essentially
identical either way.

Deliverables: results/player_ci/{figures/*.png, player_ci.parquet, ci_metrics.json}.
Run from repo root: `.venv/bin/python scripts/player_ci_bootstrap.py`.
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

C_BOOT = "#4878CF"    # bootstrap sampling band (the new, correct one)
C_MODEL = "#EE854A"   # v0 model surface band (flat in PA)
C_REF = "#8a8a8a"
MIN_PA = 100
B = 1000
SEED = 42


# ---------------------------------------------------------------------------
def pa_values(raw_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """One row per plate appearance: (batter, season, value, denom)."""
    frames = []
    for y in seasons:
        df = pl.read_parquet(raw_dir / f"statcast-{y}-slim.parquet").filter(
            pl.col("woba_denom").is_not_null()
        )
        frames.append(
            df.with_columns(
                value=pl.when(pl.col("type") == "X")
                .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
                .otherwise(pl.col("woba_value"))
            ).select("batter", season="game_year", value="value", denom="woba_denom")
        )
    return pl.concat(frames)


def bootstrap_player(value: np.ndarray, denom: np.ndarray, b: int, rng) -> dict:
    """Percentile bootstrap of xwOBA = Σvalue/Σdenom over resampled PAs."""
    n = len(value)
    point = float(value.sum() / denom.sum())
    idx = rng.integers(0, n, (b, n))
    boot = value[idx].sum(1) / denom[idx].sum(1)
    lo, hi = np.quantile(boot, [0.05, 0.95])
    # analytic sampling SE of the value-mean (denom ~ PA), for cross-checking 1/√PA
    se = float(np.std(value, ddof=1) / np.sqrt(n))
    return {"xwoba": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "width": float(hi - lo), "boot_sd": float(boot.std(ddof=1)), "se_analytic": se}


def compute_all(pav: pl.DataFrame, min_pa: int, b: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (batter, season), g in pav.group_by(["batter", "season"]):
        denom = g["denom"].to_numpy().astype(np.float64)
        if denom.sum() < min_pa:
            continue
        r = bootstrap_player(g["value"].to_numpy().astype(np.float64), denom, b, rng)
        rows.append({"batter": int(batter), "season": int(season),
                     "PA": int(round(denom.sum())), **r})
    return pl.DataFrame(rows)


def loglog_slope(pa: np.ndarray, width: np.ndarray) -> dict:
    m = (pa > 0) & (width > 0)
    s, i = np.polyfit(np.log(pa[m]), np.log(width[m]), 1)
    return {"slope": float(s), "intercept": float(i), "n": int(m.sum())}


def bin_medians(df: pl.DataFrame, col: str) -> list[dict]:
    edges = [100, 150, 200, 300, 400, 500, 750]
    out = []
    for a, c in zip(edges[:-1], edges[1:]):
        sub = df.filter((pl.col("PA") >= a) & (pl.col("PA") < c))
        if sub.height:
            out.append({"pa_mid": float(sub["PA"].median()), "median": float(sub[col].median())})
    return out


# ---------------------------------------------------------------------------
def fig_width_vs_pa(figdir: Path, df: pl.DataFrame, fit: dict,
                    boot_bins: list[dict], model_bins: list[dict]) -> None:
    pa = df["PA"].to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(pa, df["width"].to_numpy(), s=9, alpha=0.2, color=C_BOOT, edgecolors="none",
               label=f"bootstrap band ({df.height:,} player-seasons)")
    ax.plot([b["pa_mid"] for b in boot_bins], [b["median"] for b in boot_bins],
            "o-", color=C_BOOT, lw=2.4, ms=7, label="bootstrap: median width per PA bin")
    ax.plot([b["pa_mid"] for b in model_bins], [b["median"] for b in model_bins],
            "s--", color=C_MODEL, lw=2.4, ms=6, label="v0 model band: median width (flat)")
    xs = np.array([pa.min(), pa.max()])
    anchor = boot_bins[0]
    ax.plot(xs, anchor["median"] * (xs / anchor["pa_mid"]) ** -0.5,
            color=C_REF, ls=":", lw=1.8, label="ideal 1/√PA (slope −0.5)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("PA (log scale)")
    ax.set_ylabel("90% xwOBA interval width (log scale)")
    ax.set_title(f"Bootstrap band narrows with PA (slope {fit['slope']:+.2f}); v0's does not")
    ax.grid(True, which="both", color=C_REF, alpha=0.15)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(figdir / "width_vs_pa_bootstrap_vs_model.png", dpi=120)
    plt.close(fig)


def fig_forest(figdir: Path, examples: pl.DataFrame) -> None:
    """Selected players ordered by PA: bootstrap band vs v0 model band."""
    e = examples.sort("PA")
    y = np.arange(e.height)
    labels = [f"{n}  '{str(s)[2:]}  ({pa} PA)"
              for n, s, pa in zip(e["player_name"], e["season"], e["PA"])]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * e.height + 1.2))
    off = 0.16
    ax.hlines(y + off, e["ci_lo"], e["ci_hi"], color=C_BOOT, lw=3, label="bootstrap (sampling) band")
    ax.plot(e["xwoba"], y + off, "o", color=C_BOOT, ms=5)
    ax.hlines(y - off, e["xwoba_q05"], e["xwoba_q95"], color=C_MODEL, lw=3, label="v0 model (surface) band")
    ax.plot(e["xwoba_mean"], y - off, "s", color=C_MODEL, ms=5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("xwOBA")
    ax.set_title("Per-player 90% interval: bootstrap widens for small samples, v0 stays flat")
    ax.grid(True, axis="x", color=C_REF, alpha=0.2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "example_player_bands.png", dpi=120)
    plt.close(fig)


def pick_examples(df: pl.DataFrame) -> pl.DataFrame:
    """One 2024 player nearest each target PA level, to span the range cleanly."""
    d = df.filter(pl.col("season") == 2024)
    picks = []
    for target in (105, 150, 220, 320, 430, 540, 660):
        row = d.with_columns(dist=(pl.col("PA") - target).abs()).sort("dist").head(1)
        picks.append(row.drop("dist"))
    return pl.concat(picks).unique(subset=["batter", "season"]).sort("PA")


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    pav = pa_values(cfg.raw_dir, cfg.all_seasons)
    df = compute_all(pav, MIN_PA, B, SEED)
    print(f"  {df.height:,} player-seasons ({MIN_PA}+ PA)")

    # attach v0 model band + names for comparison
    pt = pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet").select(
        "batter", "season", "player_name", "xwoba_mean", "xwoba_q05", "xwoba_q95"
    ).with_columns(width_model=pl.col("xwoba_q95") - pl.col("xwoba_q05"))
    df = df.join(pt, on=["batter", "season"], how="left")

    fit = loglog_slope(df["PA"].to_numpy(), df["width"].to_numpy())
    boot_bins = bin_medians(df, "width")
    model_bins = bin_medians(df.drop_nulls("width_model"), "width_model")
    # cross-check the band really is a 1/√PA sampling band:
    m = df.drop_nulls("width_model")
    r_boot_se = float(np.corrcoef(df["width"], df["se_analytic"])[0, 1])

    outdir = cfg.results_dir / "player_ci"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    examples = pick_examples(df.drop_nulls("player_name"))
    fig_width_vs_pa(figdir, df, fit, boot_bins, model_bins)
    fig_forest(figdir, examples)
    df.write_parquet(outdir / "player_ci.parquet")

    metrics = {
        "min_pa": MIN_PA, "B": B, "n_player_seasons": df.height,
        "loglog_slope_width_vs_pa": fit,
        "bootstrap_width_bin_medians": boot_bins,
        "model_width_bin_medians": model_bins,
        "corr_bootstrap_width_vs_analytic_se": r_boot_se,
        "median_width_100_150PA": boot_bins[0]["median"],
        "median_width_500_750PA": boot_bins[-1]["median"],
        "model_median_width_100_150PA": model_bins[0]["median"],
        "model_median_width_500_750PA": model_bins[-1]["median"],
    }
    (outdir / "ci_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\n  example players:")
    with pl.Config(tbl_rows=20, fmt_str_lengths=24):
        print(examples.select("player_name", "season", "PA", "xwoba", "ci_lo", "ci_hi",
                               "width", "width_model"))
    print(f"\n  wrote {outdir}/")


if __name__ == "__main__":
    main()
