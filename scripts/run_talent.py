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
    fig.tight_layout(); fig.savefig(figdir / "shrinkage_raw_to_talent.png", dpi=120); plt.close(fig)


def fig_reliability(figdir, tbl):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tbl["PA"], tbl["reliability"], s=8, alpha=0.2, color=C_TAL)
    ax.set_xlabel("PA"); ax.set_ylabel("reliability  τ²/(τ²+SE²)")
    ax.set_title("Reliability rises with sample size")
    ax.grid(True, color=C_REF, alpha=0.15)
    fig.tight_layout(); fig.savefig(figdir / "reliability_vs_pa.png", dpi=120); plt.close(fig)


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
    (outdir / "talent_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))
    print(f"  wrote {outdir}/")


if __name__ == "__main__":
    main()
