"""Level-2 talent model runner (spec: 2026-07-19 phase2 design response; plan:
2026-07-19-xwobart-phase2-level2-talent). Stage 'l2a' proves the joint-MVN
machinery reproduces Phase 1 when restricted to xwOBA only (gates G1/G2);
stage 'full' fits the 3-D model and runs the validation races (G3-G6).
Run from repo root: `.venv/bin/python scripts/run_talent2.py [--stage l2a|full]`."""
from __future__ import annotations

import argparse
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

from src.config import load_config
from src.talent2 import (
    DIMS,
    FLOOR_SD_PER_PA,
    assemble_measurements,
    build_pa_measurements,
    build_talent2_table,
)
from benchmark_vs_savant import actual_woba, _calibrated_rmse, _pearson
from run_talent import load_pitches

C_TAL = "#4878CF"; C_TAL2 = "#EE854A"; C_REF = "#8a8a8a"

# Frozen Phase-1 anchors (results/talent/talent_metrics.json, 2026-07-18)
P1_R_PA100, P1_R_PA30, P1_N_ROWS = 0.4886, 0.4669, 2636
BOOT_B, BOOT_SEED = 500, 20260719
PREDS = ["xwoba_talent2", "xwoba_talent", "xwoba_raw", "xwoba_savant"]


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} — {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def build_measurements(cfg) -> tuple[pl.DataFrame, np.ndarray]:
    """Measurement triples + bootstrap covariances, with a row index (ridx) so
    joined/filtered frames can always index back into the S stack."""
    pam = build_pa_measurements(load_pitches(cfg, cfg.all_seasons))
    meas, S = assemble_measurements(pam, B=BOOT_B, seed=BOOT_SEED)
    return meas.with_row_index("ridx"), S


def phase1_cols(cfg) -> pl.DataFrame:
    """Frozen Phase-1 columns, renamed to avoid collisions with the L2 table."""
    p1 = pl.read_parquet(cfg.results_dir / "talent" / "talent_table.parquet")
    return p1.select(
        "batter", "season", "xwoba_talent", "xwoba_savant", "player_name",
        se2_p1="se2", p1_lo="talent_lo", p1_hi="talent_hi",
    )


def with_targets(tbl: pl.DataFrame, cfg) -> pl.DataFrame:
    act = actual_woba(cfg.raw_dir, cfg.all_seasons)
    return tbl.join(act.select("batter", "season", "actual_woba"),
                    on=["batter", "season"], how="inner")


def make_pairs(base: pl.DataFrame, seasons_t: list[int], pa_t_floor: int,
               min_pa_next: int) -> pl.DataFrame:
    """Season-T rows (PA >= pa_t_floor) joined to their T+1 actual wOBA, keeping
    players with a stable next-season sample (pa_next >= min_pa_next). Same
    logic as run_talent.validate's make_pairs, parameterized by season list so
    the select (22->23, 23->24) / confirm (24->25) split reuses it.

    The trailing sort is load-bearing, not cosmetic: polars' multi-threaded hash
    joins do not guarantee row order, so without it a fixed-seed resample draws
    different rows on every run and the paired-bootstrap CI is not reproducible.
    (batter, season_t) is unique per row, so this pins the order exactly."""
    rows = []
    for t in seasons_t:
        a = base.filter(pl.col("season") == t)
        b = base.filter(pl.col("season") == t + 1).select(
            "batter", target="actual_woba", pa_next="PA")
        rows.append(
            a.join(b, on="batter", how="inner")
            .filter((pl.col("PA") >= pa_t_floor) & (pl.col("pa_next") >= min_pa_next))
            .with_columns(season_t=pl.lit(t))
        )
    return pl.concat(rows).sort("batter", "season_t")


def race(pairs: pl.DataFrame, preds: list[str]) -> dict:
    """{pred: {r, rmse_calibrated}} + n for each predictor column. A predictor
    with nulls (xwoba_savant is a left join in the Phase-1 table) is scored on
    its own non-null subset and carries its own n, so a missing Savant value
    never silently NaNs out a whole race."""
    out = {"n": pairs.height}
    for p in preds:
        d = pairs.filter(pl.col(p).is_not_null() & pl.col("target").is_not_null())
        v, t = d[p].to_numpy(), d["target"].to_numpy()
        out[p] = {"r": _pearson(v, t), "rmse_calibrated": _calibrated_rmse(v, t)}
        if d.height != pairs.height:
            out[p]["n"] = d.height
    return out


def stage_l2a(cfg, meas: pl.DataFrame, S: np.ndarray, p1: pl.DataFrame
              ) -> tuple[pl.DataFrame, dict]:
    tbl, hypers = build_talent2_table(meas, S, dims=("xwoba",), fit_min_pa=cfg.min_pa)
    j = tbl.join(p1, on=["batter", "season"], how="inner")
    gates = [
        _gate("G1.height", tbl.height == P1_N_ROWS and j.height == tbl.height,
              f"rows {tbl.height} (phase1 {P1_N_ROWS}), joined {j.height}"),
    ]
    # G1.match is scored on the rows where the two models are SUPPOSED to agree.
    # The Level-2 variance floor (FLOOR_SD_PER_PA^2/n) deliberately overrides
    # Phase 1 on degenerate tiny samples — a hitter with 2 PA and 2 outs has zero
    # sample variance, so Phase 1's reliability is 1.0 and it reports his true
    # talent as exactly 0.000 with full confidence (results/talent/NOTES.md
    # limitation 3, which this plan set out to fix). Those rows are excluded from
    # the regression gate and reported in full under "floor_fix" below — the
    # change is counted and documented, never silently folded into a pass.
    j = j.with_columns(
        diff=(pl.col("xwoba_talent2") - pl.col("xwoba_talent")).abs(),
        floor_binds=pl.col("se2_p1") < FLOOR_SD_PER_PA ** 2 / pl.col("n"),
    )
    comp = j.filter(~pl.col("floor_binds"))
    r = _pearson(comp["xwoba_talent2"].to_numpy(), comp["xwoba_talent"].to_numpy())
    med = float(comp["diff"].median())
    gates.append(_gate("G1.match", r >= 0.999 and med <= 0.002,
                       f"corr {r:.5f}, median |diff| {med:.5f} "
                       f"on {comp.height}/{j.height} comparable rows"))
    fb = j.filter(pl.col("floor_binds"))
    floor_fix = {
        "note": "Rows where the Level-2 measurement-variance floor binds: Phase 1's "
                "analytic se2 collapsed below FLOOR_SD_PER_PA^2/n (degenerate tiny "
                "samples), so Phase 1 reported near-zero uncertainty and barely shrank "
                "them. Excluded from G1.match by design; this is the intended fix of "
                "results/talent/NOTES.md limitation 3.",
        "n_floor_bound": fb.height,
        "n_comparable": comp.height,
        "corr_all_rows": _pearson(j["xwoba_talent2"].to_numpy(),
                                  j["xwoba_talent"].to_numpy()),
        "corr_comparable": r,
        "median_pa_floor_bound": float(fb["PA"].median()),
        "max_abs_diff_floor_bound": float(fb["diff"].max()),
        "median_abs_diff_floor_bound": float(fb["diff"].median()),
        "n_phase1_se2_exactly_zero": j.filter(pl.col("se2_p1") <= 0).height,
        "biggest_moves": fb.sort("diff", descending=True).head(10).select(
            "player_name", "season", "PA", "xwoba_raw",
            phase1="xwoba_talent", level2="xwoba_talent2", moved="diff").to_dicts(),
    }
    print(f"  floor_fix: {fb.height} rows changed by the variance floor "
          f"(median PA {floor_fix['median_pa_floor_bound']:.0f}, "
          f"max move {floor_fix['max_abs_diff_floor_bound']:.4f}); "
          f"all-rows corr {floor_fix['corr_all_rows']:.4f}")
    # G2: bootstrap xwOBA SE vs Phase-1 analytic SE, rows PA>=30
    k = j.filter(pl.col("PA") >= 30)
    boot_se = np.sqrt(S[k["ridx"].to_numpy(), 0, 0])
    ana_se = np.sqrt(k["se2_p1"].to_numpy())
    r_se = _pearson(boot_se, ana_se)
    ratio = float(np.median(boot_se / ana_se))
    gates.append(_gate("G2.se", r_se >= 0.98 and 0.9 <= ratio <= 1.1,
                       f"corr {r_se:.4f}, median ratio {ratio:.3f}"))
    # G1 validation anchors: same race machinery as stage_full
    base = with_targets(j, cfg)
    r100 = race(make_pairs(base, cfg.all_seasons[:-1], cfg.min_pa, cfg.min_pa),
                ["xwoba_talent2"])["xwoba_talent2"]["r"]
    r30 = race(make_pairs(base, cfg.all_seasons[:-1], 30, cfg.min_pa),
               ["xwoba_talent2"])["xwoba_talent2"]["r"]
    gates.append(_gate("G1.val100", abs(r100 - P1_R_PA100) <= 0.005,
                       f"r {r100:.4f} vs anchor {P1_R_PA100}"))
    gates.append(_gate("G1.val30", abs(r30 - P1_R_PA30) <= 0.005,
                       f"r {r30:.4f} vs anchor {P1_R_PA30}"))
    return tbl, {"gates": gates, "hypers": hypers, "floor_fix": floor_fix,
                 "r_pa100": r100, "r_pa30": r30}


BANDS = ((30, 60), (60, 100), (100, 250), (250, 10_000_000))
WIDTH_BANDS = ((30, 60), (60, 100), (100, 250), (250, 450), (450, 10_000_000))
PAIR_BOOT_B, PAIR_BOOT_SEED = 5000, 42


def _band_label(lo: int, hi: int) -> str:
    return f"[{lo},{hi if hi < 10_000_000 else 'inf'})"


def _fast_scores(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    """(pearson r, calibrated RMSE) in closed form. For a simple OLS fit the
    residual RMSE is exactly sd(tgt)*sqrt(1-r^2) — same number _calibrated_rmse
    gets from np.polyfit, ~100x cheaper, which matters at 5,000 bootstrap reps.
    NOTE this identity also means that, for a FIXED target set, ranking by r and
    ranking by calibrated RMSE are the same ranking; G3's two criteria are not
    independent evidence. Stated explicitly in NOTES.md rather than left implied."""
    r = float(np.corrcoef(pred, tgt)[0, 1])
    return r, float(tgt.std() * np.sqrt(max(1.0 - r * r, 0.0)))


def variant_pairs(cfg, tbl_v: pl.DataFrame, p1: pl.DataFrame, act: pl.DataFrame,
                  pa_floor: int = 30, seasons_t: list[int] | None = None) -> pl.DataFrame:
    """Season-T -> T+1 pairs for any Level-2 variant table (full fit, ablation,
    zeroed-off-diagonal, restricted-hyperparameter), carrying the Phase-1
    columns so every variant races against the same frozen baseline."""
    base = (
        tbl_v.join(p1, on=["batter", "season"], how="inner")
        .join(act.select("batter", "season", "actual_woba"),
              on=["batter", "season"], how="inner")
    )
    return make_pairs(base, seasons_t or cfg.all_seasons[:-1], pa_floor, cfg.min_pa)


def paired_bootstrap(pairs: pl.DataFrame, a: str, b: str) -> dict:
    """Resample player-pair rows to get the sampling distribution of (a - b) in
    both r and calibrated RMSE. Reported, never gated — the design expects modest
    gains, so the CI is honesty about precision, not a pass bar."""
    tgt = pairs["target"].to_numpy()
    va, vb = pairs[a].to_numpy(), pairs[b].to_numpy()
    rng = np.random.default_rng(PAIR_BOOT_SEED)
    idx = rng.integers(0, len(tgt), size=(PAIR_BOOT_B, len(tgt)))
    d_r = np.empty(PAIR_BOOT_B)
    d_rmse = np.empty(PAIR_BOOT_B)
    for i, ix in enumerate(idx):
        t = tgt[ix]
        ra, ma = _fast_scores(va[ix], t)
        rb, mb = _fast_scores(vb[ix], t)
        d_r[i], d_rmse[i] = ra - rb, ma - mb
    return {
        "n_reps": PAIR_BOOT_B, "n_rows": pairs.height, "compare": f"{a} - {b}",
        "delta_r": {"mean": float(d_r.mean()),
                    "ci95": [float(np.percentile(d_r, 2.5)),
                             float(np.percentile(d_r, 97.5))],
                    "frac_better": float((d_r > 0).mean())},
        "delta_rmse_calibrated": {"mean": float(d_rmse.mean()),
                                  "ci95": [float(np.percentile(d_rmse, 2.5)),
                                           float(np.percentile(d_rmse, 97.5))],
                                  "frac_better": float((d_rmse < 0).mean())},
    }


def fig_peripheral_pull(figdir, base: pl.DataFrame):
    """The design's signature: the Level-2 correction is concentrated at low PA
    and signed by peripheral quality (high-barrel hitters pulled up).

    Two things are deliberately kept out of frame so the plot shows the effect it
    claims to show. (1) Rows on the 1-D fallback never saw a peripheral, so their
    delta is not peripheral pull. (2) Rows where the variance floor binds move for
    a different reason entirely — Phase 1's degenerate-sample bug being fixed —
    and those moves reach 0.31, which would set the y-scale and bury the real
    signal. Both are counted in l2a.floor_fix / hypers.n_fallback_1d instead.
    Colour limits are robust quantiles: barrel_rate runs to 1.0 for a hitter with
    two BBE, and a raw 0-1 scale renders every real hitter the same shade."""
    d = base.filter((pl.col("used_dims") == "3d") & (~pl.col("floor_binds"))
                    & pl.col("barrel_rate").is_not_null())
    br = d["barrel_rate"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(d["PA"], d["xwoba_talent2"] - d["xwoba_talent"], s=9,
                    alpha=0.55, c=br, cmap="viridis",
                    vmin=float(np.quantile(br, 0.02)), vmax=float(np.quantile(br, 0.98)))
    ax.axhline(0.0, color=C_REF, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("PA"); ax.set_ylabel("Level 2 − Phase 1  (xwOBA talent)")
    ax.set_title("Peripheral pull: a low-PA effect, signed by contact quality")
    fig.colorbar(sc, ax=ax, extend="both", label="barrel rate")
    ax.grid(True, color=C_REF, alpha=0.15)
    ax.text(0.99, 0.02, f"n={d.height:,} peripheral-path rows\n"
                        "(1-D fallback and floor-bound rows excluded)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C_REF)
    fig.tight_layout(); fig.savefig(figdir / "peripheral_pull.png", dpi=120); plt.close(fig)


def fig_interval_width(figdir, base: pl.DataFrame) -> list[dict]:
    """Median 90% interval width by PA band, Phase 1 vs Level 2. The plan
    expected narrowing concentrated at low PA; what the data actually shows is
    near-uniform narrowing at every band, so the title says that instead. Both
    intervals remain estimation-only (no BART surface variance) and neither has
    been coverage-validated — that is Stage 4."""
    rows = []
    for lo, hi in WIDTH_BANDS:
        b = base.filter((pl.col("PA") >= lo) & (pl.col("PA") < hi))
        if b.height < 20:
            continue
        rows.append({
            "band": _band_label(lo, hi), "n": b.height,
            "phase1": float((b["p1_hi"] - b["p1_lo"]).median()),
            "level2": float((b["talent2_hi"] - b["talent2_lo"]).median()),
        })
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w / 2, [r["phase1"] for r in rows], w, color=C_TAL, label="Phase 1 (xwOBA only)")
    ax.bar(x + w / 2, [r["level2"] for r in rows], w, color=C_TAL2, label="Level 2 (+ peripherals)")
    ax.set_xticks(x); ax.set_xticklabels([r["band"] for r in rows])
    ax.set_xlabel("PA band"); ax.set_ylabel("median 90% interval width")
    pct = [100 * (r["level2"] / r["phase1"] - 1) for r in rows]
    ax.set_title("Interval width by sample size: Level 2 narrows every band "
                 f"({min(pct):.0f}% to {max(pct):.0f}%)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", color=C_REF, alpha=0.15)
    fig.tight_layout(); fig.savefig(figdir / "interval_width_vs_pa.png", dpi=120); plt.close(fig)
    return rows


def stage_full(cfg, meas: pl.DataFrame, S: np.ndarray, p1: pl.DataFrame,
               outdir: Path, figdir: Path) -> dict:
    """The 3-D fit and the validation races that decide whether Phase 2 Stage 1
    earns its place: G3 (low-PA win, HARD), G4 (high-PA non-inferiority, HARD),
    G5 (shared-noise tripwire, diagnostic), G6 (select/confirm reporting)."""
    act = actual_woba(cfg.raw_dir, cfg.all_seasons)
    tbl2, hypers = build_talent2_table(meas, S, dims=DIMS, fit_min_pa=cfg.min_pa)
    base = (
        tbl2.join(p1, on=["batter", "season"], how="inner")
        .join(act.select("batter", "season", "actual_woba"),
              on=["batter", "season"], how="inner")
        .with_columns(floor_binds=pl.col("se2_p1") < FLOOR_SD_PER_PA ** 2 / pl.col("n"))
    )
    assert base.height == P1_N_ROWS, f"base height {base.height} != {P1_N_ROWS}"

    sel_seasons = cfg.train_seasons[:-1]          # 2022, 2023 -> model-choice split
    con_seasons = [cfg.train_seasons[-1]]         # 2024       -> confirmation only

    pairs30 = make_pairs(base, cfg.all_seasons[:-1], 30, cfg.min_pa)
    pairs100 = make_pairs(base, cfg.all_seasons[:-1], cfg.min_pa, cfg.min_pa)
    out = {
        "hypers": hypers,
        "sigma_talent_corr": (
            np.array(hypers["Sigma"]) /
            np.sqrt(np.outer(np.diag(hypers["Sigma"]), np.diag(hypers["Sigma"])))
        ).tolist(),
        "pooled_pa100": race(pairs100, PREDS),
        "pooled_pa30": race(pairs30, PREDS),
        "by_band": [],
        "split": {
            "note": "G6: model choices are scored on select (T=2022,2023); "
                    "confirm (T=2024) is reported, never tuned on.",
            "select": race(make_pairs(base, sel_seasons, 30, cfg.min_pa), PREDS),
            "confirm": race(make_pairs(base, con_seasons, 30, cfg.min_pa), PREDS),
        },
    }
    for lo, hi in BANDS:
        b = pairs30.filter((pl.col("PA") >= lo) & (pl.col("PA") < hi))
        if b.height < 20:
            continue
        out["by_band"].append({
            "band": _band_label(lo, hi),
            "median_reliability2": float(b["reliability2"].median()),
            "mean_abs_l2_minus_p1": float(
                (b["xwoba_talent2"] - b["xwoba_talent"]).abs().mean()),
            **race(b, PREDS),
        })

    # ---- HARD gates -------------------------------------------------------
    t2_30, t1_30 = out["pooled_pa30"]["xwoba_talent2"], out["pooled_pa30"]["xwoba_talent"]
    t2_100, t1_100 = out["pooled_pa100"]["xwoba_talent2"], out["pooled_pa100"]["xwoba_talent"]
    g3 = (t2_30["r"] > t1_30["r"]
          and t2_30["rmse_calibrated"] < t1_30["rmse_calibrated"])
    gates = [
        _gate("G3.lowpa", g3,
              f"PA>=30 n={out['pooled_pa30']['n']}: r {t2_30['r']:.4f} vs "
              f"{t1_30['r']:.4f} (d={t2_30['r'] - t1_30['r']:+.4f}), "
              f"rmse_cal {t2_30['rmse_calibrated']:.6f} vs "
              f"{t1_30['rmse_calibrated']:.6f} "
              f"(d={t2_30['rmse_calibrated'] - t1_30['rmse_calibrated']:+.6f})"),
        _gate("G4.highpa", t2_100["r"] >= t1_100["r"] - 0.005,
              f"PA>=100 n={out['pooled_pa100']['n']}: r {t2_100['r']:.4f} vs "
              f"{t1_100['r']:.4f} (d={t2_100['r'] - t1_100['r']:+.4f})"),
    ]
    out["gates"] = gates

    # ---- paired bootstrap (reported, not gated) ---------------------------
    out["paired_bootstrap_pa30"] = paired_bootstrap(pairs30, "xwoba_talent2", "xwoba_talent")
    pb = out["paired_bootstrap_pa30"]["delta_r"]
    print(f"  paired bootstrap PA>=30: dr mean {pb['mean']:+.4f} "
          f"CI95 [{pb['ci95'][0]:+.4f}, {pb['ci95'][1]:+.4f}] "
          f"frac_better {pb['frac_better']:.3f}")

    # ---- ablations, scored on the SELECT split only (G6) ------------------
    out["ablations"] = {"note": "Scored on select (T=2022,2023) PA>=30 pooled. "
                                "confirm reported for context; the shipped table "
                                "stays the full 3-D fit."}
    for name, dims in (("xwoba+avg_ev", ("xwoba", "avg_ev")),
                       ("xwoba+barrel_rate", ("xwoba", "barrel_rate"))):
        tv, hv = build_talent2_table(meas, S, dims=dims, fit_min_pa=cfg.min_pa)
        out["ablations"][name] = {
            "select": race(variant_pairs(cfg, tv, p1, act, 30, sel_seasons),
                           ["xwoba_talent2", "xwoba_talent"]),
            "confirm": race(variant_pairs(cfg, tv, p1, act, 30, con_seasons),
                            ["xwoba_talent2", "xwoba_talent"]),
            "n_fallback_1d": hv["n_fallback_1d"],
        }
    out["ablations"]["full_3d"] = {
        "select": race(make_pairs(base, sel_seasons, 30, cfg.min_pa),
                       ["xwoba_talent2", "xwoba_talent"]),
        "confirm": race(make_pairs(base, con_seasons, 30, cfg.min_pa),
                        ["xwoba_talent2", "xwoba_talent"]),
        "n_fallback_1d": hypers["n_fallback_1d"],
    }

    # ---- G5: shared-noise tripwire ----------------------------------------
    # Design risk #1: all three stats come from the same balls in play, so their
    # sampling errors are correlated. If we WRONGLY zero those off-diagonals and
    # the low-PA gain gets BIGGER, the gain was an artifact of fitting correlated
    # noise, not real information. Reported loudly; never silently absorbed.
    tv0, hv0 = build_talent2_table(meas, S, dims=DIMS, fit_min_pa=cfg.min_pa,
                                   zero_offdiag=True)
    r_zero = race(variant_pairs(cfg, tv0, p1, act, 30), ["xwoba_talent2", "xwoba_talent"])
    gain_proper = t2_30["r"] - t1_30["r"]
    gain_zeroed = r_zero["xwoba_talent2"]["r"] - r_zero["xwoba_talent"]["r"]
    artifact_gap = gain_zeroed - gain_proper
    alarm = bool(artifact_gap > 0.005)
    out["offdiag_tripwire"] = {
        "note": "G5: zeroing S_i's off-diagonals discards the shared sampling "
                "noise between xwOBA, EV and barrel rate. A LARGER apparent gain "
                "without them means the gain is that noise being fitted.",
        "gain_proper": gain_proper, "gain_zeroed": gain_zeroed,
        "artifact_gap": artifact_gap, "threshold": 0.005,
        "offdiag_alarm": alarm, "race_zeroed": r_zero,
    }
    print(f"  G5 tripwire: gain proper {gain_proper:+.4f}, zeroed {gain_zeroed:+.4f}, "
          f"artifact_gap {artifact_gap:+.4f} -> "
          f"{'*** ALARM ***' if alarm else 'clean'}")
    if alarm:
        print("  *** G5 ALARM: the zeroed-off-diagonal fit gains MORE than the "
              "proper fit. Treat the low-PA win as suspect until explained. ***")

    # ---- leakage sensitivity: hyperparameters from 2022-24 only ------------
    tv_s, hv_s = build_talent2_table(meas, S, dims=DIMS, fit_min_pa=cfg.min_pa,
                                     fit_seasons=cfg.train_seasons)
    r_sens = race(variant_pairs(cfg, tv_s, p1, act, 30), PREDS)
    out["hypers_2224_sensitivity"] = {
        "note": "Hyperparameters refit excluding 2025 measurement rows (posteriors "
                "still computed for all rows; 2025 talent is never a predictor, only "
                "a target). Phase-1 convention fits all seasons; this quantifies how "
                "much that convention matters.",
        "race": r_sens,
        "delta_r_vs_allseason": r_sens["xwoba_talent2"]["r"] - t2_30["r"],
        "n_fit": hv_s["n_fit"],
    }
    print(f"  leakage sensitivity: PA>=30 r {r_sens['xwoba_talent2']['r']:.4f} "
          f"vs all-season {t2_30['r']:.4f} "
          f"(d={out['hypers_2224_sensitivity']['delta_r_vs_allseason']:+.4f})")

    # ---- figures + table --------------------------------------------------
    fig_peripheral_pull(figdir, base)
    out["interval_width_by_pa"] = fig_interval_width(figdir, base)
    base.write_parquet(outdir / "talent2_table.parquet")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["l2a", "full"], default="full")
    args = ap.parse_args()
    cfg = load_config()
    outdir = cfg.results_dir / "talent2"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    meas, S = build_measurements(cfg)
    p1 = phase1_cols(cfg)

    metrics = {}
    tbl_a, metrics["l2a"] = stage_l2a(cfg, meas, S, p1)
    tbl_a.write_parquet(outdir / "l2a_table.parquet")
    if args.stage == "full":
        metrics["l2b"] = stage_full(cfg, meas, S, p1, outdir, figdir)

    (outdir / "talent2_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))
    failed = [g["name"] for st in metrics.values() for g in st["gates"] if not g["pass"]]
    print(f"  wrote {outdir}/")
    if failed:
        print(f"  HARD GATE FAILURES: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
