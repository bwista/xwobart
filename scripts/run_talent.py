"""Build the empirical-Bayes true-talent xwOBA table from the slim Statcast caches,
join display names + public Savant xwOBA, render figures, and write results/talent/.
No re-fit. Run from repo root: `.venv/bin/python scripts/run_talent.py`."""
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

from src.config import load_config
from src.talent import build_pa_values, build_talent_table

C_RAW = "#8a8a8a"; C_TAL = "#4878CF"; C_REF = "#8a8a8a"


def load_pitches(cfg, seasons) -> pl.DataFrame:
    return pl.concat([
        pl.read_parquet(cfg.raw_dir / f"statcast-{y}-slim.parquet") for y in seasons
    ])


def fig_shrinkage(figdir, tbl):
    """Raw -> talent for 2024, showing small samples pulled toward the mean."""
    d = tbl.filter(pl.col("season") == 2024)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(d["PA"], d["xwoba_raw"], s=10, alpha=0.3, color=C_RAW, label="raw xwOBA")
    ax.scatter(d["PA"], d["xwoba_talent"], s=10, alpha=0.5, color=C_TAL, label="EB true-talent")
    ax.axhline(d["mu_season"][0], color=C_REF, ls="--", lw=1, label="season mean")
    ax.set_xlabel("PA"); ax.set_ylabel("xwOBA"); ax.set_xscale("log")
    ax.set_title("Shrinkage: small samples pulled toward the mean (2024)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "shrinkage_raw_to_talent.png", dpi=200); plt.close(fig)


def fig_reliability(figdir, tbl):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tbl["PA"], tbl["reliability"], s=8, alpha=0.2, color=C_TAL)
    ax.set_xlabel("PA"); ax.set_ylabel("reliability  τ²/(τ²+SE²)")
    ax.set_title("Reliability rises with sample size")
    ax.grid(True, color=C_REF, alpha=0.15)
    fig.tight_layout(); fig.savefig(figdir / "reliability_vs_pa.png", dpi=200); plt.close(fig)


PREDS = ("xwoba_talent", "xwoba_raw", "xwoba_savant")


def validate(cfg, tbl) -> dict:
    """Predict next-season actual wOBA from {EB talent, raw xwOBA, Savant} at season T.

    Shrinkage's benefit is variance-compression across a heterogeneous-PA population,
    so it shows up in the POOLED correlation, not in within-PA-band ranking: inside a
    narrow PA band reliability is ~constant, so talent = mu(1-c)+c*raw is ~affine in raw
    and Pearson r is affine-invariant. We therefore report (1) the pooled PA>=min_pa
    result as the primary comparison against Savant's r-0.487 anchor, (2) a pooled result
    that ADMITS genuinely low-PA seasons (relax the season-T floor to 30, keep target
    >=min_pa so the T+1 wOBA is stable) — the regime shrinkage is built for, and (3) a
    by-PA-band table showing the per-band talent-vs-raw gaps are noise-dominated by
    construction (see the module note above)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from benchmark_vs_savant import actual_woba, _pearson, _calibrated_rmse

    act = actual_woba(cfg.raw_dir, cfg.all_seasons)
    base = tbl.select(
        "batter", "season", "xwoba_talent", "xwoba_raw", "xwoba_savant", "PA", "reliability"
    ).join(act.select("batter", "season", "actual_woba"), on=["batter", "season"], how="inner")

    def make_pairs(pa_t_floor: int) -> pl.DataFrame:
        """Season-T rows (PA >= pa_t_floor) joined to their T+1 actual wOBA, keeping only
        players with a stable next-season sample (pa_next >= min_pa)."""
        rows = []
        for t in cfg.all_seasons[:-1]:
            a = base.filter(pl.col("season") == t)
            b = base.filter(pl.col("season") == t + 1).select(
                "batter", target="actual_woba", pa_next="PA")
            rows.append(a.join(b, on="batter", how="inner").filter(
                (pl.col("PA") >= pa_t_floor) & (pl.col("pa_next") >= cfg.min_pa)))
        return pl.concat(rows)

    def pooled(pairs: pl.DataFrame, with_rmse: bool) -> dict:
        tgt = pairs["target"].to_numpy()
        d = {"n": pairs.height}
        for pred in PREDS:
            p = pairs[pred].to_numpy()
            d[pred] = {"r": _pearson(p, tgt)}
            if with_rmse:
                d[pred]["rmse_calibrated"] = _calibrated_rmse(p, tgt)
        d["talent_beats_raw"] = d["xwoba_talent"]["r"] > d["xwoba_raw"]["r"]
        d["talent_beats_savant"] = d["xwoba_talent"]["r"] > d["xwoba_savant"]["r"]
        return d

    primary = make_pairs(cfg.min_pa)         # PA_T >= min_pa: the anchor comparison
    inclusive = make_pairs(30)               # admit genuinely low-PA seasons
    out = {
        "note": "Pooled r is the instrument for shrinkage benefit; per-band r is "
                "affine-invariant to shrinkage and noise-dominated (see by_band).",
        "n_pairs": primary.height,
        "pooled_pa_min": {"pa_t_floor": cfg.min_pa, **pooled(primary, with_rmse=True)},
        "pooled_lowpa_inclusive": {"pa_t_floor": 30, **pooled(inclusive, with_rmse=False)},
        "by_band": [],
    }
    for lo, hi in ((30, 60), (60, 100), (100, 250), (250, 10_000_000)):
        band = inclusive.filter((pl.col("PA") >= lo) & (pl.col("PA") < hi))
        if band.height < 20:
            continue
        tgt = band["target"].to_numpy()
        out["by_band"].append({
            "band": f"[{lo},{hi if hi < 10_000_000 else 'inf'})",
            "n": band.height,
            "median_reliability": float(band["reliability"].median()),
            "mean_abs_talent_minus_raw": float(
                (band["xwoba_talent"] - band["xwoba_raw"]).abs().mean()),
            **{pred: {"r": _pearson(band[pred].to_numpy(), tgt)} for pred in PREDS},
        })
    # primary verdict vs the Savant anchor is on the PA>=min_pa pooled population
    out["beats_savant_pooled"] = out["pooled_pa_min"]["talent_beats_savant"]
    return out


def main():
    cfg = load_config()
    seasons = cfg.all_seasons
    pav = build_pa_values(load_pitches(cfg, seasons))
    tbl = build_talent_table(pav, fit_min_pa=cfg.min_pa)

    # names + public Savant xwOBA from the frozen v0 player table
    pt = pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet").select(
        "batter", "season", "player_name", "xwoba_savant"
    )
    tbl = tbl.join(pt, on=["batter", "season"], how="left")

    outdir = cfg.results_dir / "talent"; figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_shrinkage(figdir, tbl); fig_reliability(figdir, tbl)
    tbl.write_parquet(outdir / "talent_table.parquet")

    metrics = {
        "n_player_seasons": tbl.height,
        "per_season": tbl.group_by("season").agg(
            n=pl.len(), mu=pl.col("mu_season").first(), tau=pl.col("tau_season").first(),
            median_reliability=pl.col("reliability").median(),
        ).sort("season").to_dicts(),
        "biggest_shrinks": tbl.with_columns(shrink=(pl.col("xwoba_raw") - pl.col("xwoba_talent")).abs())
            .sort("shrink", descending=True).head(15)
            .select("player_name", "season", "PA", "xwoba_raw", "xwoba_talent", "reliability").to_dicts(),
    }
    metrics["validation"] = validate(cfg, tbl)
    (outdir / "talent_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))
    print(f"  wrote {outdir}/")


if __name__ == "__main__":
    main()
