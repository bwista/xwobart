"""Orchestrates v0 end to end from config. Run from the repo root:
    .venv/bin/python scripts/run_v0.py --stage A [--force-data] [--acknowledge-runtime]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ap.add_argument("--variant", choices=["v0", "spray"], default="v0",
                    help="v0 = 3 features (frozen anchor path); spray = 5 features")
    ap.add_argument("--persist-draws", type=int, default=200,
                    help="thinned per-event value draws to persist (spray variant only)")
    ap.add_argument("--marginalize-spray", type=int, default=0, metavar="M",
                    help="spray-marginalized per-event values via M equal-mass league "
                         "quantiles per (EV, LA, stand) cell; 0 = skip. Costs M x the "
                         "prediction pass (~30 min at M=9 on Stage C)")
    ap.add_argument("--m-trees", type=int, default=None, metavar="M",
                    help="override the stage's m_trees (capacity experiments). Fit cost "
                         "is ~linear in this: 50 -> ~27 min, 200 -> ~107 min at Stage C")
    ap.add_argument("--tag", default="", metavar="STR",
                    help="suffix for the output dir, e.g. --tag m200a -> "
                         "results/stage_C_spray_m200a. Keeps capacity runs from "
                         "overwriting the frozen anchor dirs")
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--acknowledge-runtime", action="store_true",
                    help="user has signed off on a >30 min fit estimate")
    args = ap.parse_args()

    t_start = time.perf_counter()
    cfg = load_config()
    stage = cfg.stages[args.stage]
    if args.m_trees is not None:
        stage = dataclasses.replace(stage, m_trees=args.m_trees)
    # results/stage_C is NEVER overwritten by the spray run: the ELPD anchor lives there.
    suffix = "" if args.variant == "v0" else f"_{args.variant}"
    suffix += f"_{args.tag}" if args.tag else ""
    stage_dir = cfg.results_dir / f"stage_{args.stage}{suffix}"
    features = prep.VARIANT_FEATURES[args.variant]
    figdir = stage_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"stage": args.stage, "seed": cfg.seed, "kit_sha": kit_sha(),
                     "variant": args.variant, "features": features,
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
    if args.variant == "spray":
        # Imputation table from TRAINING seasons only, applied to both, so the holdout
        # never imputes itself. hc missingness is ~0.04% of BBE; rows are imputed and
        # flagged, NEVER dropped -- the holdout must stay at 122,006 events for the
        # -80107 ELPD anchor to be comparable.
        cell, hand = prep.spray_impute_table(bbe_train)
        bbe_train = prep.add_spray(bbe_train, cell, hand)
        bbe_hold = prep.add_spray(bbe_hold, cell, hand)
        metrics["hc_imputed_rate"] = {
            "train": float(bbe_train["hc_imputed"].mean()),
            "holdout": float(bbe_hold["hc_imputed"].mean()),
        }
        assert bbe_hold.height == 122006, f"holdout BBE moved: {bbe_hold.height}"
    dist = prep.class_distribution(bbe_train)
    print(f"train class distribution:\n{dist}")
    metrics["class_distribution_train"] = dist.to_dicts()
    non_bbe = prep.build_non_bbe_pa(pl.concat([df_train, df_hold]))

    # ---- fit ----
    fit_df = prep.stratified_subsample(bbe_train, stage.subsample, cfg.seed)
    X_fit, y_fit = prep.build_features(fit_df, features)
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
    if args.variant == "spray":
        # save_idata writes only the InferenceData; the fitted trees live on the live op
        # (src/model.py:106) and are otherwise unrecoverable, which is why the fit and the
        # marginalization must share one process. Pickling them demotes a failed 60-minute
        # run to a 30-minute one. Best-effort ONLY: this sits immediately after the most
        # expensive step in the plan, so an unpicklable tree object must not kill the run.
        try:
            # all_trees is a multiprocessing Manager ListProxy (pymc_bart/bart.py:143),
            # not a list. Pickling the proxy stores only a ~200-byte connection token
            # that raises FileNotFoundError on load once the manager process is gone --
            # caught by the Step 7.4 smoke. Materialize it first.
            trees = list(mdl["mu"].owner.op.all_trees)
            with open(stage_dir / "all_trees.pkl", "wb") as f:
                pickle.dump(trees, f, protocol=4)
            print(f"all_trees.pkl: {len(trees)} draws, "
                  f"{(stage_dir / 'all_trees.pkl').stat().st_size / 1e6:.1f} MB")
        except Exception as exc:
            print(f"WARN: could not pickle all_trees ({type(exc).__name__}: {exc}); "
                  f"a re-run would need a full refit")

    warnings = model_mod.sanity_check(idata, seed=cfg.seed)
    metrics["sanity_warnings"] = warnings
    # Collapsed class probabilities are a real failure -> hard stop at every stage.
    if any("collapsed" in w for w in warnings):
        sys.exit(f"STOP (spec §7.4 — collapsed class probabilities): {warnings}")
    # BART mu-cell R-hat is structurally high: the sum-of-trees is not identified at the
    # individual-cell level, so this is NOT a meaningful convergence signal. The quantities
    # we actually use (class probabilities / expected values) are gated by
    # verify_oos_mechanism below (corr ~1.0 in practice). Per the Stage A review it is a
    # warning, not a hard stop, at every stage (deviation from the plan's hard B/C gate).
    if warnings:
        print(f"WARN (BART mu R-hat is structural, not a convergence stop): {warnings}")

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
    Xtr, ytr = prep.build_features(pt_train, features)
    Xho, yho = prep.build_features(pt_hold, features)
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
    del ev_all          # ~970 MB; spray marginalization allocates another 727 MB per pass

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
    # Durability: the thesis metric is on disk before any optional work runs.
    (stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    if args.variant == "v0":
        s_grid, X_g, X_b = evaluate.contact_grids(cfg.sprint_grid)
        b_g = model_mod.predict_and_reduce(mdl, idata, X_g, None, w, cfg, cfg.seed)
        b_b = model_mod.predict_and_reduce(mdl, idata, X_b, None, w, cfg, cfg.seed)
        grounder_ev, barrel_ev = b_g.ev_draws, b_b.ev_draws
    else:
        s_grid, grids = evaluate.contact_grids(cfg.sprint_grid, variant="spray")
        gd = {k: model_mod.predict_and_reduce(mdl, idata, X, None, w, cfg, cfg.seed).ev_draws
              for k, X in grids.items()}
        grounder_ev, barrel_ev = gd["grounder_pull"], gd["barrel_pull"]
        # E7: sprint speed's payoff should concentrate on PULLED grounders
        slopes = {k: float(np.polyfit(s_grid, v.mean(axis=0), 1)[0]) for k, v in gd.items()}
        metrics["sprint_migration"] = {
            "slopes_per_ftps": slopes,
            "pull_minus_oppo": slopes["grounder_pull"] - slopes["grounder_oppo"],
        }
        print("sprint migration:", metrics["sprint_migration"])
    gb_mask = (pt_hold["bb_type"] == "ground_ball").to_numpy()
    metrics["localization"] = evaluate.localization(
        figdir, s_grid, grounder_ev, barrel_ev,
        pt_hold["sprint_speed"].to_numpy(),
        pt_hold["launch_speed_angle"].fill_null(-1).to_numpy(),
        ev_mean_hold,
    )

    if args.variant == "spray":
        # Spec §"Recommended spec": PDP/importance for spray (HR band in LA x spray).
        # Needs the live model, so it cannot be deferred to a later plan.
        la_ax, sp_ax, X_pdp = evaluate.la_spray_grid()
        p_pdp = model_mod.predict_and_reduce(
            mdl, idata, X_pdp, None, w, cfg, cfg.seed).p_mean      # (n, K)
        hr = p_pdp[:, prep.CLASS_NAMES.index("home_run")].reshape(len(la_ax), len(sp_ax))
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(sp_ax, la_ax, hr, shading="nearest", cmap="viridis")
        fig.colorbar(im, ax=ax, label="P(home run)")
        ax.set_xlabel("spray_pull (deg; POSITIVE = pulled)")
        ax.set_ylabel("launch_angle (deg)")
        ax.set_title("HR band in LA x spray (RHB, EV 103 mph)")
        fig.tight_layout()
        fig.savefig(figdir / "pdp_la_spray_hr.png", dpi=120)
        plt.close(fig)
        metrics["pdp_hr_band"] = {
            "max_p_hr": float(hr.max()),
            "argmax_la": float(la_ax[hr.max(axis=1).argmax()]),
            "argmax_spray": float(sp_ax[hr.max(axis=0).argmax()]),
        }
        print("HR band peak:", metrics["pdp_hr_band"])

    metrics["variable_importance"] = evaluate.variable_importance(figdir, mdl, idata, X_fit,
                                                                 features)
    actual_gb = pt_hold["woba_value"].to_numpy().astype(np.float64)[gb_mask]
    metrics["undercorrection_gb_holdout"] = evaluate.undercorrection(
        actual_gb, ev_mean_hold[gb_mask], pub_hold[gb_mask],
        pt_hold["sprint_speed"].to_numpy()[gb_mask],
    )

    # Per-event holdout log-likelihood, persisted for EVERY variant (it is <1 MB). Two
    # runs over the same holdout can then be compared with a PAIRED per-event test, whose
    # standard error is far below the ~244 on each total because both models are driven by
    # the same EV/LA signal and their per-event errors are strongly correlated. The digest
    # is what makes pairing safe: it proves the two runs scored the same events in the same
    # order rather than merely the same count.
    np.save(stage_dir / "lppd_i_holdout.npy", b_hold.lppd_i.astype(np.float64))
    metrics["holdout_order_digest"] = hashlib.sha256(
        pt_hold["batter"].to_numpy().tobytes()).hexdigest()[:16]

    # ---- optional Stage-4 products (AFTER the ELPD verdict is durable on disk) ----
    if args.variant == "spray":
        # Stage-4 prerequisite: per-event value draws. Stage 4 folds their between-draw
        # variance into S_i[0,0], which is what finally makes interval coverage testable.
        # Persisting these is the whole reason the refit is worth doing once, not twice.
        def persist(tag: str, ev: np.ndarray, pt: pl.DataFrame) -> dict:
            k = min(args.persist_draws, ev.shape[0])
            idx = np.linspace(0, ev.shape[0] - 1, k).astype(int)
            arr = np.ascontiguousarray(ev[idx], dtype=np.float32)
            # Assert BEFORE writing 291 MB, not after.
            assert arr.shape[1] == pt.height, "draw columns must align with key rows"
            np.save(stage_dir / f"ev_draws_{tag}.npy", arr)
            pt.select("batter", season=pl.col("game_year"),
                      woba_denom=pl.col("woba_denom"),
                      launch_speed=pl.col("launch_speed"),
                      launch_angle=pl.col("launch_angle"),
                      spray_pull=pl.col("spray_pull"), stand_R=pl.col("stand_R"),
                      sprint_speed=pl.col("sprint_speed"),
                      hc_imputed=pl.col("hc_imputed")
                      ).with_row_index("row").write_parquet(
                          stage_dir / f"ev_draws_keys_{tag}.parquet")
            # The assert above is near-tautological (both derive from the same frame);
            # what actually matters is row ORDER. Stamp a checkable key so Stage 4 can
            # detect a reordering rather than trusting the contract.
            return {"shape": list(arr.shape), "mb": round(arr.nbytes / 1e6, 1),
                    "draw_index": idx.tolist(),
                    "batter_order_digest": hashlib.sha256(
                        pt["batter"].to_numpy().tobytes()).hexdigest()[:16]}

        metrics["persisted_draws"] = {
            "train": persist("train", b_train.ev_draws, pt_train),
            "holdout": persist("holdout", b_hold.ev_draws, pt_hold),
        }
        print("persisted draws:", metrics["persisted_draws"])

    if args.variant == "spray" and args.marginalize_spray:
        # Design risk 2: conditioning a player rollup on per-ball direction credits spray
        # LUCK. The marginalized value replaces v(x_e) with its league average over spray
        # given EV x LA x stand -- no refit, just M extra prediction passes.
        M = args.marginalize_spray
        qs = np.linspace(0.5 / M, 1 - 0.5 / M, M)          # M equal-mass quantiles
        si = features.index("spray_pull")
        src = bbe_train.with_columns(
            _ev=(pl.col("launch_speed") // prep.SPRAY_EV_BIN).cast(pl.Int32),
            _la=(pl.col("launch_angle") // prep.SPRAY_LA_BIN).cast(pl.Int32))
        qt = (src.group_by("_ev", "_la", "stand_R")
                 .agg([pl.col("spray_pull").quantile(q).alias(f"q{i}")
                       for i, q in enumerate(qs)], _n=pl.len())
                 .filter(pl.col("_n") >= prep.SPRAY_MIN_CELL).drop("_n")
                 .sort("_ev", "_la", "stand_R"))     # polars group_by is not order-stable
        marg, sparse_rate = {}, {}
        for tag, pt in (("train", pt_train), ("holdout", pt_hold)):
            j = (pt.with_row_index("_r").with_columns(
                     _ev=(pl.col("launch_speed") // prep.SPRAY_EV_BIN).cast(pl.Int32),
                     _la=(pl.col("launch_angle") // prep.SPRAY_LA_BIN).cast(pl.Int32))
                   .join(qt, on=["_ev", "_la", "stand_R"], how="left").sort("_r"))
            assert j.height == pt.height, "marginalization join changed row count"
            Q = j.select([f"q{i}" for i in range(M)]).to_numpy()          # (n, M)
            base = j.select(features).to_numpy().astype(np.float64)       # (n, 5)
            # Capture sparsity BEFORE the fill -- after np.where, Q is all-finite and the
            # rate would be identically 0.0, silently killing the no-op diagnostic that
            # Steps 7.4 and 8.1 depend on.
            sparse_rate[tag] = float(np.mean(~np.isfinite(Q)))
            # cells too sparse for quantiles keep their OBSERVED spray (identity)
            Q = np.where(np.isfinite(Q), Q, base[:, si:si + 1])
            acc = np.zeros(base.shape[0])
            for m in range(M):
                Xm = base.copy()
                Xm[:, si] = Q[:, m]
                acc += model_mod.predict_and_reduce(
                    mdl, idata, Xm, None, w, cfg, cfg.seed).ev_draws.mean(axis=0)
            marg[tag] = acc / M
            np.save(stage_dir / f"ev_marginalized_{tag}.npy", marg[tag].astype(np.float32))
        metrics["marginalize_spray"] = {
            "M": M,
            "sparse_cell_rate": sparse_rate,
            "mean_abs_shift_train": float(np.abs(
                marg["train"] - b_train.ev_draws.mean(axis=0)).mean()),
        }
        print("marginalized:", metrics["marginalize_spray"])

    # ---- report ----
    total_s = time.perf_counter() - t_start
    metrics["total_runtime_s"] = total_s
    (stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    rep, cal, el, un = (metrics["replication"], metrics["calibration"],
                        metrics["elpd"], metrics["undercorrection_gb_holdout"])
    update_results_md(cfg.results_dir, f"{args.stage}{suffix}", [
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
