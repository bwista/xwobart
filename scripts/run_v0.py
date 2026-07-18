"""Orchestrates v0 end to end from config. Run from the repo root:
    .venv/bin/python scripts/run_v0.py --stage A [--force-data] [--acknowledge-runtime]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src import data, evaluate, model as model_mod, prep, rollup
from src.config import load_config


def kit_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", "../kinferencetoolkit", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def update_results_md(results_dir: Path, stage: str, lines: list[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "RESULTS.md"
    start, end = f"<!-- stage_{stage} -->", f"<!-- /stage_{stage} -->"
    block = "\n".join([start, f"## Stage {stage}", *lines, end])
    text = path.read_text() if path.exists() else "# xwobart v0 results\n"
    if start in text:
        text = text.split(start)[0] + block + text.split(end)[1]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def estimate_minutes(cfg, stage, n_rows: int) -> float | None:
    """Extrapolate fit runtime from the most recent completed stage's metrics
    (cost ~ rows x trees x (tune + draws) x chains)."""
    for prev in ("B", "A"):
        mpath = cfg.results_dir / f"stage_{prev}" / "metrics.json"
        if prev != stage.name and mpath.exists():
            m = json.loads(mpath.read_text())
            if "fit_runtime_s" in m and "fit_cost_units" in m:
                cost = n_rows * stage.m_trees * (stage.tune + stage.draws) * stage.chains
                return m["fit_runtime_s"] * cost / m["fit_cost_units"] / 60
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--acknowledge-runtime", action="store_true",
                    help="user has signed off on a >30 min fit estimate")
    args = ap.parse_args()

    t_start = time.perf_counter()
    cfg = load_config()
    stage = cfg.stages[args.stage]
    stage_dir = cfg.results_dir / f"stage_{args.stage}"
    figdir = stage_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"stage": args.stage, "seed": cfg.seed, "kit_sha": kit_sha(),
                     "stage_settings": vars(stage) | {}}

    # ---- data ----
    data.build_season_caches(cfg, force=args.force_data)
    metrics["coverage_gaps"] = json.loads((cfg.raw_dir / "coverage.json").read_text())
    df_train = data.load_seasons(cfg, cfg.train_seasons)
    df_hold = data.load_seasons(cfg, [cfg.holdout_season])
    sprint = data.fetch_sprint_speed(cfg)
    expected = data.fetch_expected_stats(cfg)

    # ---- prep ----
    def prep_bbe(df: pl.DataFrame, tag: str) -> pl.DataFrame:
        bbe, drop_report = prep.filter_bbe(df)
        bbe, imp_rate = data.merge_sprint_speed(bbe, sprint)
        bbe = prep.add_outcome_class(bbe)
        print(f"[{tag}] drops:\n{drop_report}\nimputation:\n{imp_rate}")
        metrics[f"drop_report_{tag}"] = drop_report.to_dicts()
        metrics[f"imputation_{tag}"] = imp_rate.to_dicts()
        return bbe

    bbe_train = prep_bbe(df_train, "train")
    bbe_hold = prep_bbe(df_hold, "holdout")
    dist = prep.class_distribution(bbe_train)
    print(f"train class distribution:\n{dist}")
    metrics["class_distribution_train"] = dist.to_dicts()
    non_bbe = prep.build_non_bbe_pa(pl.concat([df_train, df_hold]))

    # ---- fit ----
    fit_df = prep.stratified_subsample(bbe_train, stage.subsample, cfg.seed)
    X_fit, y_fit = prep.build_features(fit_df)
    metrics["fit_rows"] = int(X_fit.shape[0])
    metrics["subsample_seed"] = cfg.seed
    est = estimate_minutes(cfg, stage, X_fit.shape[0])
    if est is not None:
        print(f"estimated fit runtime: ~{est:.0f} min")
        metrics["fit_runtime_estimate_min"] = est
        if est > 30 and not args.acknowledge_runtime:
            sys.exit(f"Estimated fit ~{est:.0f} min (> 30). Get user sign-off, then "
                     f"re-run with --acknowledge-runtime (spec §13).")
    mdl, idata, fit_s = model_mod.fit(X_fit, y_fit, stage, cfg.seed)
    print(f"fit runtime: {fit_s / 60:.1f} min")
    metrics["fit_runtime_s"] = fit_s
    metrics["fit_cost_units"] = X_fit.shape[0] * stage.m_trees * (stage.tune + stage.draws) * stage.chains
    model_mod.save_idata(idata, stage_dir / "idata.nc")

    warnings = model_mod.sanity_check(idata, seed=cfg.seed)
    metrics["sanity_warnings"] = warnings
    if any("collapsed" in w for w in warnings):
        sys.exit(f"STOP (spec §7.4): {warnings}")
    if warnings and args.stage in ("B", "C"):
        sys.exit(f"STOP (spec §7.4 — R-hat gate is hard for stages B/C): {warnings}")
    if warnings:
        print(f"WARN (tolerated at Stage A only): {warnings}")

    metrics["oos_verification"] = model_mod.verify_oos_mechanism(mdl, idata, X_fit, cfg, cfg.seed)
    if not metrics["oos_verification"]["pass"]:
        sys.exit(f"STOP: OOS mechanism failed verification: {metrics['oos_verification']}")

    # ---- weights + prediction (train uses OOS path too: the fit saw only the subsample) ----
    w, w_warn = rollup.linear_weights(bbe_train)
    metrics["linear_weights"] = dict(zip(prep.CLASS_NAMES, np.round(w, 4).tolist()))
    metrics["weight_warnings"] = w_warn
    print("linear weights:", metrics["linear_weights"], w_warn or "")

    def capped(df: pl.DataFrame) -> pl.DataFrame:
        if stage.predict_cap is not None and df.height > stage.predict_cap:
            return prep.stratified_subsample(df, stage.predict_cap, cfg.seed)
        return df

    pt_train, pt_hold = capped(bbe_train), capped(bbe_hold)
    metrics["predict_rows"] = {"train": pt_train.height, "holdout": pt_hold.height,
                               "capped": stage.predict_cap is not None}
    Xtr, ytr = prep.build_features(pt_train)
    Xho, yho = prep.build_features(pt_hold)
    b_train = model_mod.predict_and_reduce(mdl, idata, Xtr, ytr, w, cfg, cfg.seed)
    b_hold = model_mod.predict_and_reduce(mdl, idata, Xho, yho, w, cfg, cfg.seed)

    # ---- rollup ----
    keys = pl.concat([
        pt_train.select("batter", season=pl.col("game_year"), woba_denom=pl.col("woba_denom")),
        pt_hold.select("batter", season=pl.col("game_year"), woba_denom=pl.col("woba_denom")),
    ])
    ev_all = np.concatenate([b_train.ev_draws, b_hold.ev_draws], axis=1)
    table = rollup.build_player_table(rollup.player_rollup(ev_all, keys, non_bbe), expected)
    table.write_parquet(stage_dir / "player_table.parquet")

    # ---- the four checks ----
    ev_mean_train = b_train.ev_draws.mean(axis=0).astype(np.float64)
    ev_mean_hold = b_hold.ev_draws.mean(axis=0).astype(np.float64)
    pub_train = pt_train["estimated_woba_using_speedangle"].to_numpy().astype(np.float64)
    pub_hold = pt_hold["estimated_woba_using_speedangle"].to_numpy().astype(np.float64)
    metrics["replication"] = evaluate.replication(
        figdir,
        ev_mean_train=ev_mean_train, public_train=pub_train,
        ev_mean_hold=ev_mean_hold, public_hold=pub_hold,
        ls_hold=pt_hold["launch_speed"].to_numpy(), la_hold=pt_hold["launch_angle"].to_numpy(),
        player_table=table, min_pa=cfg.min_pa, train_seasons=cfg.train_seasons, seed=cfg.seed,
    )
    metrics["calibration"] = evaluate.calibration(figdir, b_hold.p_mean.astype(np.float64),
                                                  yho, cfg.reliability_bins)
    metrics["elpd"] = evaluate.elpd_metrics(b_hold.lppd_i, b_hold.meanlog_i)

    s_grid, X_g, X_b = evaluate.contact_grids(cfg.sprint_grid)
    b_g = model_mod.predict_and_reduce(mdl, idata, X_g, None, w, cfg, cfg.seed)
    b_b = model_mod.predict_and_reduce(mdl, idata, X_b, None, w, cfg, cfg.seed)
    gb_mask = (pt_hold["bb_type"] == "ground_ball").to_numpy()
    metrics["localization"] = evaluate.localization(
        figdir, s_grid, b_g.ev_draws, b_b.ev_draws,
        pt_hold["sprint_speed"].to_numpy(),
        pt_hold["launch_speed_angle"].fill_null(-1).to_numpy(),
        ev_mean_hold,
    )
    metrics["variable_importance"] = evaluate.variable_importance(figdir, mdl, idata, X_fit)
    actual_gb = pt_hold["woba_value"].to_numpy().astype(np.float64)[gb_mask]
    metrics["undercorrection_gb_holdout"] = evaluate.undercorrection(
        actual_gb, ev_mean_hold[gb_mask], pub_hold[gb_mask],
        pt_hold["sprint_speed"].to_numpy()[gb_mask],
    )

    # ---- report ----
    total_s = time.perf_counter() - t_start
    metrics["total_runtime_s"] = total_s
    (stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    rep, cal, el, un = (metrics["replication"], metrics["calibration"],
                        metrics["elpd"], metrics["undercorrection_gb_holdout"])
    update_results_md(cfg.results_dir, args.stage, [
        f"- kit_sha: {metrics['kit_sha']} | seed: {cfg.seed} | fit rows: {metrics['fit_rows']}"
        f" | predict rows: {metrics['predict_rows']}",
        f"- coverage gaps: {metrics['coverage_gaps'] or 'none'}",
        f"- fit runtime: {fit_s / 60:.1f} min | total: {total_s / 60:.1f} min",
        f"- linear weights: {metrics['linear_weights']}",
        f"- volumes/drops per season: train {metrics['drop_report_train']};"
        f" holdout {metrics['drop_report_holdout']}",
        f"- sprint imputation rates: train {metrics['imputation_train']};"
        f" holdout {metrics['imputation_holdout']}",
        f"- replication r — event train {rep['event_r_train']:.3f}, event holdout"
        f" {rep['event_r_holdout']:.3f}, player train {rep['player_r_train']:.3f},"
        f" player holdout {rep['player_r_holdout']:.3f}",
        f"- calibration — weighted ECE {cal['ece_weighted']:.4f}",
        f"- ELPD (lppd) {el['elpd_lppd']:.1f} ± {el['elpd_se']:.1f} over {el['n_events']} events",
        f"- undercorrection corr — model {un['model_residual_sprint_corr']:.3f} vs public"
        f" {un['public_residual_sprint_corr']:.3f}",
        f"- localization slopes (per ft/s) — grounder"
        f" {metrics['localization']['grounder_slope_per_ftps']:.4f}, barrel"
        f" {metrics['localization']['barrel_slope_per_ftps']:.4f}",
        f"- sanity warnings: {warnings or 'none'}",
    ])
    print(f"stage {args.stage} complete in {total_s / 60:.1f} min -> {stage_dir}")


if __name__ == "__main__":
    main()
