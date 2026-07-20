"""Stage 3 gate E8: spray-CONDITIONED vs spray-MARGINALIZED player rollup, raced
against next-season actual wOBA. Run from repo root:
    .venv/bin/python scripts/rollup_ab.py

Design risk 2: conditioning a player rollup on per-ball direction credits spray LUCK.
Public spray-adjusted xwOBA variants typically describe the same season better without
predicting the next one better, so the conditioned rollup is NOT assumed to win -- the
race decides, and a difference under ~0.001 calibrated RMSE at n~1,000 pairs is a tie.

Pure analysis over the arrays Task 8 persisted: no model, no idata, no refit. Level 2
still consumes PUBLIC Savant per-event xwOBA, so pushing the winner through talent2 is
Stage 4's job; what Stage 3 settles is which rollup predicts next season better.
Plan: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from benchmark_vs_savant import _pearson, actual_woba
from run_talent2 import BANDS, make_pairs, paired_bootstrap, race
from src import data, prep, rollup
from src.config import load_config

PREDS = ["xwoba_conditioned", "xwoba_marginalized", "xwoba_v0", "xwoba_savant"]
# Palette validated with the dataviz six-checks validator (CVD dE 13.8, normal-vision
# 17.3, all-pairs). The two model rollups are ONE entity in two configurations, so they
# take two lightness steps of the repo's model hue rather than two competing hues --
# four separable categorical hues do not exist alongside the fixed Savant orange.
# The neutral and the light step read low-chroma BY DESIGN (recessive baseline; same
# entity), which is why every bar also carries a direct value label.
C = {"xwoba_conditioned": "#2C4E8A", "xwoba_marginalized": "#8DB4E2",
     "xwoba_v0": "#767676", "xwoba_savant": "#EE854A"}
LABEL = {"xwoba_conditioned": "conditioned", "xwoba_marginalized": "marginalized",
         "xwoba_v0": "v0 (3-feature)", "xwoba_savant": "Savant"}


def load_side(stage_dir: Path, tag: str) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Keys + conditioned draws + marginalized values for one split.

    The alignment contract is POSITIONAL (axis 1 of the .npy <-> row order of the key
    parquet) and an off-by-one here is silent, so assert before doing anything else."""
    keys = pl.read_parquet(stage_dir / f"ev_draws_keys_{tag}.parquet")
    cond = np.load(stage_dir / f"ev_draws_{tag}.npy")            # (S, n)
    marg = np.load(stage_dir / f"ev_marginalized_{tag}.npy")     # (n,)
    assert cond.shape[-1] == keys.height, f"{tag}: draws/keys misaligned"
    assert marg.shape[-1] == keys.height, f"{tag}: marginalized/keys misaligned"
    return keys, cond, marg


def build_rollups(cfg, stage_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    k_tr, d_tr, m_tr = load_side(stage_dir, "train")
    k_ho, d_ho, m_ho = load_side(stage_dir, "holdout")
    # Concatenate keys and arrays in the SAME order so each player-season is rolled up
    # once. `row` restarts at 0 in each parquet, so it would carry duplicates after the
    # concat and quietly undercut the positional contract -- drop it (player_rollup
    # needs only batter, season, woba_denom).
    keys = pl.concat([k_tr, k_ho]).drop("row")
    cond = np.concatenate([d_tr, d_ho], axis=1)
    marg = np.concatenate([m_tr, m_ho])
    assert cond.shape[1] == marg.size == keys.height, "concatenated arrays misaligned"

    non_bbe = prep.build_non_bbe_pa(data.load_seasons(cfg, cfg.all_seasons))
    roll_cond = rollup.player_rollup(cond, keys, non_bbe)
    # A 1-draw stack: player_rollup handles S == 1 (it zeroes xwoba_sd) and only the
    # mean is needed for the race.
    roll_marg = rollup.player_rollup(marg[None, :], keys, non_bbe)
    shapes = {"n_events": int(keys.height), "draws": int(cond.shape[0]),
              "train_events": int(k_tr.height), "holdout_events": int(k_ho.height),
              "player_seasons": int(roll_cond.height)}
    return roll_cond, roll_marg, shapes


def build_base(cfg, roll_cond: pl.DataFrame, roll_marg: pl.DataFrame) -> pl.DataFrame:
    """One row per (batter, season): PA, actual wOBA and the four predictors.

    Select-and-rename BEFORE joining. All three frames collide on PA / xwoba_mean (both
    rollups emit PA, xwoba_mean, xwoba_sd, xwoba_q05, xwoba_q95; the frozen v0 table
    emits PA, xwoba_mean, xwoba_savant) and polars would silently suffix `_right`.
    PA comes from the rollup (capital, as make_pairs expects); dropping actual_woba's
    lowercase `pa` sidesteps the rename entirely."""
    cond_t = roll_cond.select("batter", "season", "PA", xwoba_conditioned="xwoba_mean")
    marg_t = roll_marg.select("batter", "season", xwoba_marginalized="xwoba_mean")
    v0_t = (pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet")
              .select("batter", "season", "xwoba_savant", xwoba_v0="xwoba_mean"))
    act_t = actual_woba(cfg.raw_dir, cfg.all_seasons).select(
        "batter", "season", "actual_woba")
    return (cond_t.join(marg_t, on=["batter", "season"], how="inner")
                  .join(v0_t, on=["batter", "season"], how="inner")
                  .join(act_t, on=["batter", "season"], how="inner")
                  .sort("batter", "season"))


def fig_rmse_by_band(figdir: Path, by_band: list[dict]) -> None:
    """Grouped bars, next-season calibrated RMSE per PA band, four predictors.

    Every bar carries a direct value label: two palette slots sit below 3:1 contrast on
    white, which the dataviz contrast check says obligates visible labels (not
    dismissable), and labels double as the secondary encoding for identity."""
    fig, ax = plt.subplots(figsize=(9, 5))
    n, w = len(PREDS), 0.20
    x = np.arange(len(by_band))
    for i, p in enumerate(PREDS):
        v = np.array([b[p]["rmse_calibrated"] * 1e3 for b in by_band])
        off = (i - (n - 1) / 2) * w
        # 2px surface gap between adjacent bars (w * 0.92) per the mark spec.
        ax.bar(x + off, v, w * 0.92, color=C[p], label=LABEL[p], zorder=3)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom",
                    fontsize=7, color="#3d3d3d", zorder=4)
    ax.set_xticks(x, [b["band"] for b in by_band])
    ax.set_xlabel("season-T PA band")
    ax.set_ylabel(r"next-season calibrated RMSE ($\times 10^{-3}$; lower is better)")
    ax.set_title("Spray-conditioned vs spray-marginalized rollup — next-season wOBA")
    ax.grid(axis="y", color="#e4e4e4", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(figdir / "next_season_rmse_by_band.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    stage_dir = cfg.results_dir / "stage_C_spray"
    outdir = cfg.results_dir / "rollup_ab"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading persisted draws + marginalized values")
    roll_cond, roll_marg, shapes = build_rollups(cfg, stage_dir)
    print(f"  {shapes}")

    print("[2/4] building the race base")
    base = build_base(cfg, roll_cond, roll_marg)
    print(f"  base rows {base.height}")

    print("[3/4] racing against next-season actual wOBA")
    seasons_t = cfg.all_seasons[:-1]
    pairs30 = make_pairs(base, seasons_t, 30, cfg.min_pa)
    pairs100 = make_pairs(base, seasons_t, cfg.min_pa, cfg.min_pa)
    out: dict = {
        "shapes": shapes,
        "pooled_pa30": race(pairs30, PREDS),
        "pooled_pa100": race(pairs100, PREDS),
        "by_band": [],
        "by_pair": [],
        # RMSE is primary and r secondary by the plan -- but note _fast_scores' identity
        # (calibrated RMSE = sd(target) * sqrt(1 - r^2)): for a FIXED target set the two
        # rank predictors identically, so they are not independent evidence.
        "note_rmse_vs_r": "for a fixed target set, ranking by calibrated RMSE and by r "
                          "is the same ranking (rmse = sd(target)*sqrt(1-r^2))",
    }
    for lo, hi in BANDS:
        p = pairs30.filter((pl.col("PA") >= lo) & (pl.col("PA") < hi))
        if p.height < 25:
            continue
        out["by_band"].append({"band": f"{lo}-{hi}" if hi < 10_000 else f"{lo}+",
                               "lo": lo, "hi": hi, **race(p, PREDS)})
    for t in seasons_t:
        p = pairs30.filter(pl.col("season_t") == t)
        out["by_pair"].append({"pair": f"{t}->{t + 1}", **race(p, PREDS)})

    out["paired_bootstrap_pa30"] = paired_bootstrap(
        pairs30, "xwoba_conditioned", "xwoba_marginalized")
    out["paired_bootstrap_pa100"] = paired_bootstrap(
        pairs100, "xwoba_conditioned", "xwoba_marginalized")

    # Descriptive side: the design predicts an INVERSION -- conditioned describes the
    # same season better while marginalized predicts the next one better. Both are
    # legitimate products; only the PREDICTIVE winner designates Stage 4's talent input.
    same = base.filter((pl.col("PA") >= cfg.min_pa) & pl.col("xwoba_savant").is_not_null())
    sav = same["xwoba_savant"].to_numpy()
    out["same_season_savant_corr_pa100"] = {
        "n": same.height,
        **{p: _pearson(same[p].to_numpy(), sav) for p in PREDS if p != "xwoba_savant"},
    }

    d = out["paired_bootstrap_pa30"]["delta_rmse_calibrated"]
    rm30 = {p: out["pooled_pa30"][p]["rmse_calibrated"] for p in PREDS}
    rm100 = {p: out["pooled_pa100"][p]["rmse_calibrated"] for p in PREDS}
    winner30 = min(("xwoba_conditioned", "xwoba_marginalized"), key=rm30.get)
    winner100 = min(("xwoba_conditioned", "xwoba_marginalized"), key=rm100.get)
    tie = abs(rm30["xwoba_conditioned"] - rm30["xwoba_marginalized"]) < 1e-3
    out["verdict"] = {
        "pa30_lower_rmse": winner30, "pa100_lower_rmse": winner100,
        "delta_rmse_pa30_cond_minus_marg": rm30["xwoba_conditioned"] - rm30["xwoba_marginalized"],
        "resolvable": not tie,
        "stage4_talent_input": ("tie - under 0.001 calibrated RMSE at this n"
                                if tie else winner30),
    }

    print("[4/4] figure + write")
    fig_rmse_by_band(figdir, out["by_band"])
    (outdir / "rollup_ab_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    (roll_cond.select("batter", "season", "PA", xwoba_conditioned="xwoba_mean")
        .join(roll_marg.select("batter", "season", xwoba_marginalized="xwoba_mean"),
              on=["batter", "season"], how="inner")
        .sort("batter", "season")
        .write_parquet(outdir / "marginalized_values.parquet"))

    print(f"\n  pooled PA>=30  calibrated RMSE: "
          f"{ {k.replace('xwoba_', ''): round(v, 6) for k, v in rm30.items()} }")
    print(f"  pooled PA>=100 calibrated RMSE: "
          f"{ {k.replace('xwoba_', ''): round(v, 6) for k, v in rm100.items()} }")
    print(f"  paired bootstrap PA>=30 (conditioned - marginalized) RMSE: "
          f"mean {d['mean']:+.6f} CI95 [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] "
          f"frac_better {d['frac_better']:.3f}")
    print(f"  same-season corr vs Savant (PA>=100): {out['same_season_savant_corr_pa100']}")
    print(f"  VERDICT: {out['verdict']}")
    print(f"  wrote {outdir}/")


if __name__ == "__main__":
    main()
