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
    the select (22->23, 23->24) / confirm (24->25) split reuses it."""
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
    return pl.concat(rows)


def race(pairs: pl.DataFrame, preds: list[str]) -> dict:
    """{pred: {r, rmse_calibrated}} + n for each predictor column."""
    tgt = pairs["target"].to_numpy()
    out = {"n": pairs.height}
    for p in preds:
        v = pairs[p].to_numpy()
        out[p] = {"r": _pearson(v, tgt), "rmse_calibrated": _calibrated_rmse(v, tgt)}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["l2a", "full"], default="l2a")
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
        raise SystemExit("stage 'full' arrives in Task 6")   # replaced in Task 6

    (outdir / "talent2_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))
    failed = [g["name"] for st in metrics.values() for g in st["gates"] if not g["pass"]]
    print(f"  wrote {outdir}/")
    if failed:
        print(f"  HARD GATE FAILURES: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
