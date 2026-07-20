"""Stage 2 sign QC: prove the pull-relative spray mirror is not backwards, BEFORE
spending 27 minutes on the surface refit. Run from repo root:
    .venv/bin/python scripts/qc_spray.py

Getting the mirror wrong silently reflects half the league and would poison the Stage-3
fit invisibly. Every single-fault mirror error (forgot to mirror / mirrored both hands /
mirrored the wrong hand / swapped the atan2 arguments) trips at least one HARD gate here.
Plan: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md"""
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

from src import prep
from src.config import load_config

# 2024 named anchors (id, stand, comparator, threshold). Frozen IDs, not names: the
# name resolver is a network/cache dependency and must not be able to fail a gate.
NAMED_2024 = [
    (656941, "Kyle Schwarber",   "L", "gt", 8.0),    # measured +13.90
    (670623, "Isaac Paredes",    "R", "gt", 8.0),    # measured +11.88
    (663624, "Ryan Mountcastle", "R", "lt", 0.0),    # measured  -5.16 (oppo)
]
C_L, C_R = "#4878CF", "#EE854A"


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} - {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    cfg = load_config()
    outdir = cfg.results_dir / "stage2_rebuild"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    raw = {y: pl.read_parquet(cfg.raw_dir / f"statcast-{y}-slim.parquet")
           for y in cfg.all_seasons}
    bbe = {y: prep.filter_bbe(df)[0] for y, df in raw.items()}
    # Imputation table from the TRAINING seasons only; applied to every season.
    cell, hand = prep.spray_impute_table(pl.concat([bbe[y] for y in cfg.train_seasons]))
    sp = {y: prep.add_spray(b, cell, hand) for y, b in bbe.items()}

    gates: list[dict] = []
    per_season: dict[int, dict] = {}
    for y, d in sp.items():
        obs = d.filter(~pl.col("hc_imputed"))
        g = (obs.group_by("stand").agg(mean_pull=pl.col("spray_pull").mean(), n=pl.len())
                .sort("stand"))
        hr = (obs.filter(pl.col("events") == "home_run")
                 .group_by("stand").agg(mean_pull=pl.col("spray_pull").mean(),
                                        frac_pos=(pl.col("spray_pull") > 0).mean(),
                                        n=pl.len()).sort("stand"))
        per_season[y] = {
            "n_bbe": d.height,
            "hc_imputed_rate": float(d["hc_imputed"].mean()),
            "abs_phi_gt_45": float((obs["phi_raw"].abs() > 45).mean()),
            "mean_pull": dict(zip(g["stand"], g["mean_pull"])),
            "hr_mean_pull": dict(zip(hr["stand"], hr["mean_pull"])),
            "hr_frac_pull_side": dict(zip(hr["stand"], hr["frac_pos"])),
            "hr_n": dict(zip(hr["stand"], hr["n"])),
        }

    def _all(key: str, cmp) -> bool:
        return all(len(v[key]) == 2 and all(cmp(x) for x in v[key].values())
                   for v in per_season.values())

    gates.append(_gate("S1.league_mean_pull_positive", _all("mean_pull", lambda x: x > 1.0),
                       str({y: {k: round(v, 2) for k, v in d["mean_pull"].items()}
                            for y, d in per_season.items()})))
    gates.append(_gate("S2.hr_mean_pull", _all("hr_mean_pull", lambda x: x >= 12.0),
                       str({y: {k: round(v, 1) for k, v in d["hr_mean_pull"].items()}
                            for y, d in per_season.items()})))
    gates.append(_gate("S3.hr_frac_pull_side", _all("hr_frac_pull_side", lambda x: x >= 0.70),
                       str({y: {k: round(v, 3) for k, v in d["hr_frac_pull_side"].items()}
                            for y, d in per_season.items()})))

    d24 = sp[2024].filter(~pl.col("hc_imputed"))
    agg = (d24.group_by("batter", "stand").agg(mp=pl.col("spray_pull").mean(), n=pl.len())
              .filter(pl.col("n") >= 250).sort("batter", "stand"))
    named, ok_named = {}, True
    for bid, nm, st, how, thr in NAMED_2024:
        row = agg.filter((pl.col("batter") == bid) & (pl.col("stand") == st))
        val = float(row["mp"][0]) if row.height else float("nan")
        good = (val > thr) if how == "gt" else (val < thr)
        named[nm] = {"stand": st, "mean_pull": val, "want": f"{how} {thr}", "pass": bool(good)}
        ok_named &= bool(good)
    gates.append(_gate("S4.named_anchors", ok_named, json.dumps(named, default=float)))

    # S5: the mirror must be resolved PER EVENT, not per player-season. Counting switch
    # hitters would be vacuous (no bug in add_spray can change that count). Instead split
    # the switch hitters' own rows by stand: under a modal-hand mirror the minority-hand
    # subgroup flips negative while every league-level gate above still passes.
    switch_ids = (sp[2024].group_by("batter").agg(k=pl.col("stand").n_unique())
                          .filter(pl.col("k") > 1)["batter"].to_list())   # list, not Series:
                                                    # is_in on a same-dtype Series is deprecated
    sw = (d24.filter(pl.col("batter").is_in(switch_ids))
             .group_by("stand").agg(mean_pull=pl.col("spray_pull").mean(), n=pl.len())
             .sort("stand"))
    sw_ok = sw.height == 2 and bool((sw["mean_pull"] > 0).all()) and len(switch_ids) >= 1
    gates.append(_gate("S5.stand_is_per_event", sw_ok,
                       f"{len(switch_ids)} switch batters (expect ~65); by stand "
                       f"{dict(zip(sw['stand'], [round(v, 2) for v in sw['mean_pull']]))}"))
    gates.append(_gate("S6.imputation_rate",
                       all(v["hc_imputed_rate"] < 0.001 for v in per_season.values()),
                       str({y: round(v["hc_imputed_rate"] * 100, 4) for y, v in per_season.items()})))

    # Figure 1: pull-relative spray density by hand (both must peak on the positive side)
    fig, ax = plt.subplots(figsize=(7, 5))
    for st, c in (("L", C_L), ("R", C_R)):
        v = d24.filter(pl.col("stand") == st)["spray_pull"].to_numpy()
        ax.hist(v, bins=90, range=(-90, 90), density=True, histtype="step",
                color=c, lw=1.6, label=f"{st}HB (mean {v.mean():+.2f} deg)")
    ax.axvline(0, color="#8a8a8a", ls="--", lw=1)
    ax.set_xlabel("spray_pull (deg; POSITIVE = pulled)"); ax.set_ylabel("density")
    ax.set_title("2024 pull-relative spray - both hands must lean positive")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "spray_by_hand.png", dpi=120); plt.close(fig)

    # Figure 2: raw direction on home runs (the asymmetry BART needs `stand` to see)
    fig, ax = plt.subplots(figsize=(7, 5))
    hr24 = d24.filter(pl.col("events") == "home_run")
    for st, c in (("L", C_L), ("R", C_R)):
        v = hr24.filter(pl.col("stand") == st)["phi_raw"].to_numpy()
        ax.hist(v, bins=60, range=(-60, 60), density=True, histtype="step",
                color=c, lw=1.6, label=f"{st}HB HR (mean {v.mean():+.1f} deg)")
    ax.axvline(0, color="#8a8a8a", ls="--", lw=1)
    ax.set_xlabel("phi_raw (deg; negative = LEFT field)"); ax.set_ylabel("density")
    ax.set_title("2024 home runs, RAW direction - opposite peaks by hand")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "spray_hr_raw_direction.png", dpi=120); plt.close(fig)

    (outdir / "spray_qc.json").write_text(json.dumps(
        {"per_season": per_season, "named_2024": named,
         "n_switch_2024": len(switch_ids), "switch_by_stand": sw.to_dicts(),
         "gates": gates}, indent=2, default=float))
    failed = [g["name"] for g in gates if not g["pass"]]
    print(f"  wrote {outdir}/spray_qc.json and 2 figures")
    if failed:
        print(f"\n  HARD GATE FAILURES: {failed}")
        print("  DO NOT RUN THE STAGE-3 FIT. The mirror is wrong; check src/prep._spray_cols:")
        print("    - RHB pull to LEFT, so spray_pull = -phi_raw for stand == 'R'")
        print("    - phi_raw = atan2(hc_x - 125.42, 198.27 - hc_y): x-term FIRST")
        raise SystemExit(1)
    print("\n  Sign QC PASS. Safe to spend 27 minutes on the refit.")


if __name__ == "__main__":
    main()
