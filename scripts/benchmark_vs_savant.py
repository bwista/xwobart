"""Head-to-head predictive benchmark: v0 model xwOBA vs public Savant xwOBA.

The canonical xwOBA-validity test, NO re-fit. An estimator of a hitter's true
talent is "more accurate" if its year-T value predicts year-(T+1) *actual* wOBA
better. We race three predictors of actual wOBA_{T+1}:

  * model    -- v0 posterior-mean xwOBA_T  (results/stage_C/player_table.parquet)
  * savant   -- public Savant xwOBA_T      (est_woba, same table's xwoba_savant)
  * naive    -- actual wOBA_T              (the baseline xwOBA must beat)

Actual wOBA_s per (batter, season) is rebuilt from the slim caches as
sum(woba_value)/sum(woba_denom). Pairs span 2022->23, 23->24, 24->25.

Primary metric: Pearson r vs actual wOBA_{T+1} (scale-invariant, the standard in
these studies). Also calibrated RMSE (residual std after an OLS rescale, so a
predictor is not penalised for a constant scale/bias offset) and raw RMSE. A paired
bootstrap over player-pairs gives a CI on (r_model - r_savant).

Caveat (stated in the note): for the model, year-T is an IN-SAMPLE season (its BBE
surface was fit on 2022-24); there is no pair whose predictor-year is holdout (2025
has no T+1). Savant likewise uses year-T data for its year-T estimate, so the
comparison is symmetric, but this is retrodiction->prediction, not pure OOS.
A clean OOS event-level head-to-head needs a re-fit (the fitted BART trees are not
persisted in idata.nc) -- recommended as a follow-up that also saves per-event EVs.

Run from repo root: `.venv/bin/python scripts/benchmark_vs_savant.py`.
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

C_MODEL = "#4878CF"   # v0 model (categorical slot 1)
C_SAVANT = "#EE854A"  # Savant   (categorical slot 2)
C_NAIVE = "#8a8a8a"   # naive actual (neutral baseline)
PRED_COLOR = {"model": C_MODEL, "savant": C_SAVANT, "naive": C_NAIVE}
PRED_LABEL = {"model": "v0 model xwOBA", "savant": "Savant xwOBA", "naive": "actual wOBA (naive)"}

MIN_PA = 100


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def actual_woba(raw_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Actual wOBA per (batter, season) = sum(woba_value)/sum(woba_denom)."""
    frames = []
    for y in seasons:
        p = raw_dir / f"statcast-{y}-slim.parquet"
        df = pl.read_parquet(p).filter(pl.col("woba_denom").is_not_null())
        frames.append(
            df.group_by("batter", "game_year").agg(
                woba_num=pl.col("woba_value").sum(),
                pa=pl.col("woba_denom").sum(),
            )
        )
    return (
        pl.concat(frames)
        .with_columns(actual_woba=pl.col("woba_num") / pl.col("pa"))
        .rename({"game_year": "season"})
        .select("batter", "season", "pa", "actual_woba")
    )


def build_pairs(base: pl.DataFrame, seasons: list[int], min_pa: int) -> pl.DataFrame:
    """Self-join season T with T+1 on batter; keep 100+ PA in both years.
    Columns: batter, season_t, model, savant, naive (=actual_t), target (=actual_{t+1})."""
    rows = []
    for t in seasons[:-1]:
        a = base.filter(pl.col("season") == t)
        b = base.filter(pl.col("season") == t + 1).select("batter", target="actual_woba", pa_next="PA")
        j = a.join(b, on="batter", how="inner").filter(
            (pl.col("PA") >= min_pa) & (pl.col("pa_next") >= min_pa)
        )
        rows.append(j.select(
            "batter",
            season_t=pl.lit(t),
            model=pl.col("xwoba_mean"),
            savant=pl.col("xwoba_savant"),
            naive=pl.col("actual_woba"),
            target=pl.col("target"),
        ))
    return pl.concat(rows)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _calibrated_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Residual std after OLS(target ~ pred): the RMSE achievable once the
    predictor is optimally rescaled, so scale/bias offsets don't distort it."""
    s, b = np.polyfit(pred, target, 1)
    return float(np.sqrt(np.mean((target - (s * pred + b)) ** 2)))


def score(pairs: pl.DataFrame) -> dict:
    target = pairs["target"].to_numpy()
    out = {}
    for pred in ("model", "savant", "naive"):
        p = pairs[pred].to_numpy()
        out[pred] = {
            "r": _pearson(p, target),
            "r2": _pearson(p, target) ** 2,
            "rmse_calibrated": _calibrated_rmse(p, target),
            "rmse_raw": float(np.sqrt(np.mean((p - target) ** 2))),
        }
    return out


def bootstrap_gap(pairs: pl.DataFrame, seed: int = 42, n_boot: int = 5000) -> dict:
    """Paired bootstrap over player-pairs: distribution of r_model - r_savant."""
    model = pairs["model"].to_numpy()
    savant = pairs["savant"].to_numpy()
    target = pairs["target"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(target)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps[i] = _pearson(model[idx], target[idx]) - _pearson(savant[idx], target[idx])
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return {
        "r_model_minus_r_savant": _pearson(model, target) - _pearson(savant, target),
        "ci95": [float(lo), float(hi)],
        "frac_model_better": float(np.mean(gaps > 0)),
    }


# ---------------------------------------------------------------------------
def fig_bars(figdir: Path, pooled: dict, per_pair: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    preds = ["model", "savant", "naive"]
    # pooled Pearson r
    ax1.bar(range(3), [pooled[p]["r"] for p in preds],
            color=[PRED_COLOR[p] for p in preds], width=0.6)
    for i, p in enumerate(preds):
        ax1.text(i, pooled[p]["r"] + 0.004, f"{pooled[p]['r']:.3f}", ha="center", fontsize=10)
    ax1.set_xticks(range(3)); ax1.set_xticklabels([PRED_LABEL[p] for p in preds], fontsize=9)
    ax1.set_ylabel("Pearson r vs next-season actual wOBA")
    ax1.set_ylim(0, max(pooled[p]["r"] for p in preds) * 1.15)
    ax1.set_title("Predictive accuracy (pooled, all season-pairs)")
    ax1.grid(True, axis="y", color=C_NAIVE, alpha=0.2)

    # per-pair grouped
    pairs = sorted(per_pair.keys())
    x = np.arange(len(pairs)); w = 0.26
    for j, p in enumerate(preds):
        ax2.bar(x + (j - 1) * w, [per_pair[k][p]["r"] for k in pairs], width=w,
                color=PRED_COLOR[p], label=PRED_LABEL[p])
    ax2.set_xticks(x); ax2.set_xticklabels(pairs)
    ax2.set_xlabel("season pair (predict T+1 from T)")
    ax2.set_ylabel("Pearson r")
    ax2.set_title("By season pair")
    ax2.grid(True, axis="y", color=C_NAIVE, alpha=0.2)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "predictive_accuracy.png", dpi=200)
    plt.close(fig)


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    seasons = cfg.all_seasons

    pt = pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet")
    act = actual_woba(cfg.raw_dir, seasons)
    base = pt.select("batter", "season", "xwoba_mean", "xwoba_savant", "PA").join(
        act.select("batter", "season", "actual_woba"), on=["batter", "season"], how="inner"
    )

    pairs = build_pairs(base, seasons, MIN_PA)
    print(f"  {pairs.height} player-pairs ({MIN_PA}+ PA both years)")

    pooled = score(pairs)
    per_pair = {}
    for t in seasons[:-1]:
        sub = pairs.filter(pl.col("season_t") == t)
        if sub.height >= 30:
            per_pair[f"{t}->{t+1}"] = score(sub)
    gap = bootstrap_gap(pairs)

    verdict = (
        "model beats Savant" if gap["ci95"][0] > 0 else
        "Savant beats model" if gap["ci95"][1] < 0 else
        "model ~ Savant (no significant gap)"
    )

    metrics = {
        "min_pa": MIN_PA,
        "n_pairs": pairs.height,
        "n_per_pair": {f"{t}->{t+1}": pairs.filter(pl.col("season_t") == t).height
                       for t in seasons[:-1]},
        "pooled": pooled,
        "per_pair": per_pair,
        "model_vs_savant_bootstrap": gap,
        "verdict": verdict,
    }

    outdir = cfg.results_dir / "benchmark"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_bars(figdir, pooled, per_pair)
    (outdir / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    pairs.write_parquet(outdir / "pairs.parquet")

    print(json.dumps(metrics, indent=2))
    print(f"\n  VERDICT: {verdict}")
    print(f"  wrote {outdir}/")


if __name__ == "__main__":
    main()
