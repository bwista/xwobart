"""Stage 2: rebuild the slim Statcast caches with hc_x/hc_y/stand, behind a hard
reproduction gate (R1-R6). Run from repo root:
    .venv/bin/python scripts/rebuild_caches.py

The gate exists because build_season_caches(force=True) re-reads the UPSTREAM monthly
cache, which KIT may have updated since these caches were built. Adding columns cannot
change existing rows; an upstream data revision silently would -- and every frozen
anchor in this repo (2,636 player-seasons; r 0.4886/0.4669; ELPD -80107 over 122,006
events) depends on those rows being unchanged.
Plan: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src import data, prep
from src.config import load_config
from src.talent import build_pa_values, build_talent_table

# The 15 columns that existed BEFORE this stage. R2 compares exactly these.
PRE_REBUILD_COLUMNS = data.KEEP_COLUMNS[:15]
# Frozen anchors (results/stage_C/metrics.json, results/talent*/; measured 2026-07-19)
BBE_PER_SEASON = {2022: 118891, 2023: 122070, 2024: 122634, 2025: 122006}
P1_ROWS = 2636
L2_R_PA30, L2_R_PA100 = 0.469817, 0.490783     # full precision: a 4-dp anchor eats
                                               # 17% of a 1e-4 tolerance on rounding alone


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} - {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    cfg = load_config()
    outdir = cfg.results_dir / "stage2_rebuild"
    outdir.mkdir(parents=True, exist_ok=True)
    backup = cfg.raw_dir / "prerebuild"
    backup.mkdir(parents=True, exist_ok=True)
    paths = {y: cfg.raw_dir / f"statcast-{y}-slim.parquet" for y in cfg.all_seasons}
    gates: list[dict] = []

    print("[1/5] fingerprinting + backing up the existing caches")
    before = {y: data.cache_fingerprint(p, PRE_REBUILD_COLUMNS) for y, p in paths.items()}
    for y, p in paths.items():
        shutil.copy2(p, backup / p.name)
    print(f"  backup -> {backup}")

    print("[2/5] rebuilding (force=True) - this re-reads the upstream monthly cache")
    data.build_season_caches(cfg, force=True)

    print("[3/5] reproduction gates")
    after = {y: data.cache_fingerprint(p, PRE_REBUILD_COLUMNS) for y, p in paths.items()}
    gates.append(_gate("R1.rows",
                       all(before[y]["n_rows"] == after[y]["n_rows"] for y in paths),
                       str({y: (before[y]["n_rows"], after[y]["n_rows"]) for y in paths})))
    gates.append(_gate("R2.digest",
                       all(before[y]["digest"] == after[y]["digest"] for y in paths),
                       str({y: (before[y]["digest"], after[y]["digest"]) for y in paths})))

    new_cols = {}
    for y, p in paths.items():
        df = pl.read_parquet(p, columns=["type", "hc_x", "hc_y", "stand"])
        x = df.filter(pl.col("type") == "X")
        new_cols[y] = {
            "stand_null_rate_bbe": float(x["stand"].is_null().mean()),
            "hc_null_rate_bbe": float(
                (x["hc_x"].is_null() | x["hc_y"].is_null()).mean()),
        }
    gates.append(_gate("R3.new_columns",
                       all(v["stand_null_rate_bbe"] == 0.0 and v["hc_null_rate_bbe"] < 0.001
                           for v in new_cols.values()), str(new_cols)))

    print("[4/5] downstream reproduction")
    pitches = pl.concat([pl.read_parquet(paths[y]) for y in cfg.all_seasons])
    p1_new = build_talent_table(build_pa_values(pitches), fit_min_pa=cfg.min_pa)
    p1_old = pl.read_parquet(cfg.results_dir / "talent" / "talent_table.parquet")
    j = p1_new.select("batter", "season", t_new="xwoba_talent").join(
        p1_old.select("batter", "season", t_old="xwoba_talent"),
        on=["batter", "season"], how="inner").sort("batter", "season")
    dmax = float((j["t_new"] - j["t_old"]).abs().max()) if j.height else float("inf")
    gates.append(_gate("R4.phase1",
                       p1_new.height == P1_ROWS and j.height == P1_ROWS and dmax < 1e-12,
                       f"rows {p1_new.height} (want {P1_ROWS}), joined {j.height}, "
                       f"max|delta| {dmax:.3e}"))

    bbe_counts = {}
    for y in cfg.all_seasons:
        b, _ = prep.filter_bbe(pl.read_parquet(paths[y]))
        bbe_counts[y] = b.height
    gates.append(_gate("R5.bbe_counts", bbe_counts == BBE_PER_SEASON,
                       f"{bbe_counts} vs frozen {BBE_PER_SEASON}"))

    # Persist R1-R5 BEFORE the Level-2 subprocess. The rebuild is already destructive by
    # this point; if step 5 raises, the evidence for the gates that DID run must survive.
    report = {"before": before, "after": after, "new_columns": new_cols,
              "bbe_counts": bbe_counts, "gates": gates}
    (outdir / "rebuild_report.json").write_text(json.dumps(report, indent=2, default=float))

    print("[5/5] re-running the Level-2 talent model against the rebuilt caches")
    root = Path(__file__).resolve().parents[1]
    rc = subprocess.run([sys.executable, "scripts/run_talent2.py", "--stage", "full"],
                        capture_output=True, text=True, cwd=root)
    print(rc.stdout[-3000:] or rc.stderr[-3000:])
    m2 = json.loads((cfg.results_dir / "talent2" / "talent2_metrics.json").read_text())
    # Key path verified against the shipped results/talent2/talent2_metrics.json:
    # l2b -> {hypers, sigma_talent_corr, pooled_pa100, pooled_pa30, by_band, split,
    #         gates, paired_bootstrap_pa30, ablations, offdiag_tripwire, ...}
    r30 = m2["l2b"]["pooled_pa30"]["xwoba_talent2"]["r"]
    r100 = m2["l2b"]["pooled_pa100"]["xwoba_talent2"]["r"]
    gates.append(_gate("R6.level2",
                       rc.returncode == 0 and abs(r30 - L2_R_PA30) < 5e-4
                       and abs(r100 - L2_R_PA100) < 5e-4,
                       f"exit {rc.returncode}, r30 {r30:.6f} (want {L2_R_PA30}), "
                       f"r100 {r100:.6f} (want {L2_R_PA100})"))

    report |= {"level2": {"r30": r30, "r100": r100}, "gates": gates}
    (outdir / "rebuild_report.json").write_text(json.dumps(report, indent=2, default=float))
    failed = [g["name"] for g in gates if not g["pass"]]
    print(f"  wrote {outdir}/rebuild_report.json")
    if failed:
        print(f"\n  HARD GATE FAILURES: {failed}")
        print("  The upstream data moved -- this is NOT a code bug. Do not proceed to")
        print("  Stage 3; every frozen anchor is invalid until reconciled. Restore with:")
        print(f"    cp {backup}/statcast-*-slim.parquet {cfg.raw_dir}/")
        raise SystemExit(1)
    print("\n  All reproduction gates PASS. Caches now carry hc_x/hc_y/stand.")


if __name__ == "__main__":
    main()
