"""Does spray help once the tree budget is matched? (Stage-3 follow-up.)

Run from the repo root. Takes ~5.3 hours at the default m_trees=200; safe to stop and
re-run -- completed fits are skipped:

    .venv/bin/python scripts/capacity_experiment.py            # full run
    .venv/bin/python scripts/capacity_experiment.py --dry-run  # print the plan, fit nothing
    .venv/bin/python scripts/capacity_experiment.py --analyze  # verdict from existing fits
    .venv/bin/python scripts/capacity_experiment.py --m-trees 100   # cheaper rung

WHY. Stage 3's gate E1 failed: the 5-feature spray surface scored -79,876.3 against the
frozen 3-feature anchor's -80,107.5, a +231-nat gain where +1000 was required -- and
+231 sits INSIDE the +267-nat run-to-run noise floor measured the same day. Against a
same-session v0 replicate the spray surface was 35.9 nats WORSE.

But the model plainly LEARNED spray: it ranks #3 of 5 in variable importance and the
HR-band PDP climbs 0.20 -> 0.37 from opposite-field to pulled. The leading explanation is
CAPACITY DILUTION rather than missing information -- at a fixed m_trees=50 the same tree
budget is spread over 5 dimensions instead of 3, so splits spent resolving spray are
splits no longer spent resolving EV x LA. This experiment tests exactly that by giving
BOTH variants the same larger budget and comparing them to each other, with a fresh
anchor, instead of comparing spray to a frozen 3-feature number.

THREE FITS, all at the same m_trees:
  1. v0    (3 features)  -> the new anchor at this capacity
  2. v0    (3 features)  -> a REPLICATE, whose gap to fit 1 is the null noise floor here
  3. spray (5 features)  -> the treatment

Two verdicts are computed. The UNPAIRED one repeats Stage 3's logic (total ELPD vs the
new anchor, judged against the newly measured noise floor). The PAIRED one is the sharper
test and the reason run_v0 now persists per-event holdout log-likelihood for every
variant: over the same 122,006 events in the same order, it bootstraps the per-event
difference spray - v0. Both models are driven mostly by the same EV/LA signal, so their
per-event errors are strongly correlated and the paired standard error is far below the
~244 on each total -- it can resolve differences the unpaired test cannot see.

Plan of record: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md
Stage-3 result: results/RESULTS.md, section "Phase 2 Stage 3".
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

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BOOT_B, BOOT_SEED = 5000, 42
# Stage-3 reference points, for context in the final report only -- nothing is gated on
# them, because this experiment establishes its own anchor at its own capacity.
M50 = {"v0_anchor": -80107.495, "v0_replicate": -79840.4, "spray": -79876.3,
       "noise_floor": 267.1, "m_trees": 50}


def fits(m: int, stage: str = "C") -> list[dict]:
    return [
        {"key": "v0_a", "variant": "v0", "tag": f"m{m}a",
         "dir": f"stage_{stage}_m{m}a", "what": "new anchor at this capacity"},
        {"key": "v0_b", "variant": "v0", "tag": f"m{m}b",
         "dir": f"stage_{stage}_m{m}b", "what": "replicate -> the null noise floor"},
        {"key": "spray", "variant": "spray", "tag": f"m{m}",
         "dir": f"stage_{stage}_spray_m{m}", "what": "treatment (5 features)"},
    ]


def run_one(cfg, f: dict, m: int, stage: str) -> bool:
    """Run one fit unless it already finished. Returns True if usable afterwards."""
    out = cfg.results_dir / f["dir"]
    done = (out / "metrics.json").exists() and (out / "lppd_i_holdout.npy").exists()
    if done:
        print(f"  [skip] {f['key']}: {out} already complete")
        return True
    cmd = [sys.executable, "scripts/run_v0.py", "--stage", stage,
           "--variant", f["variant"], "--m-trees", str(m), "--tag", f["tag"],
           "--acknowledge-runtime"]
    print(f"  [run ] {f['key']} ({f['what']})\n         {' '.join(cmd)}")
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=ROOT)
    mins = (time.perf_counter() - t0) / 60
    if rc.returncode != 0:
        print(f"  [FAIL] {f['key']} exited {rc.returncode} after {mins:.1f} min")
        return False
    print(f"  [done] {f['key']} in {mins:.1f} min")
    return True


def load(cfg, f: dict) -> dict | None:
    p = cfg.results_dir / f["dir"] / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    m["_lppd_i"] = cfg.results_dir / f["dir"] / "lppd_i_holdout.npy"
    return m


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Bootstrap the per-event difference (a - b) over matched holdout events.

    Resampling EVENTS (not fits) measures how much of the total gap is carried by a
    reproducible shift in per-event fit rather than by a handful of events."""
    d = a - b
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, d.size, size=(BOOT_B, d.size))
    sums = d[idx].sum(axis=1)
    return {
        "total_delta_nats": float(d.sum()),
        "mean_per_event": float(d.mean()),
        "paired_se_nats": float(np.sqrt(d.size) * d.std(ddof=1)),
        "ci95_nats": [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))],
        "frac_events_better": float((d > 0).mean()),
        "n_events": int(d.size),
    }


def analyze(cfg, m: int, stage: str = "C") -> dict:
    F = {f["key"]: f for f in fits(m, stage)}
    got = {k: load(cfg, f) for k, f in F.items()}
    missing = [k for k, v in got.items() if v is None]
    if missing:
        print(f"\nCannot analyze — missing fits: {missing}")
        return {}

    e = {k: v["elpd"]["elpd_lppd"] for k, v in got.items()}
    n = {k: v["elpd"]["n_events"] for k, v in got.items()}
    assert len(set(n.values())) == 1, f"holdout sizes differ: {n} — not comparable"
    digests = {k: v.get("holdout_order_digest") for k, v in got.items()}
    paired_ok = len(set(digests.values())) == 1 and all(digests.values())

    noise = abs(e["v0_a"] - e["v0_b"])
    out = {
        "m_trees": m, "n_events": n["v0_a"], "elpd": e,
        "noise_floor_nats": noise,
        "spray_minus_v0a": e["spray"] - e["v0_a"],
        "spray_minus_v0b": e["spray"] - e["v0_b"],
        "spray_minus_v0_mean": e["spray"] - 0.5 * (e["v0_a"] + e["v0_b"]),
        "holdout_order_digests": digests, "paired_valid": paired_ok,
        "stage3_m50_reference": M50,
        "ece": {k: v["calibration"]["ece_weighted"] for k, v in got.items()},
        "fit_minutes": {k: round(v["fit_runtime_s"] / 60, 1) for k, v in got.items()},
    }

    print("\n" + "=" * 72)
    print(f"CAPACITY EXPERIMENT — m_trees = {m}, holdout {n['v0_a']:,} events")
    print("=" * 72)
    print(f"  v0 fit A (anchor)     {e['v0_a']:12.1f}")
    print(f"  v0 fit B (replicate)  {e['v0_b']:12.1f}")
    print(f"  spray (5 features)    {e['spray']:12.1f}")
    print(f"\n  null noise floor |A-B|        {noise:+.1f} nats")
    print(f"  spray - v0(A)                 {out['spray_minus_v0a']:+.1f} nats")
    print(f"  spray - v0(B)                 {out['spray_minus_v0b']:+.1f} nats")
    print(f"  spray - mean(v0)              {out['spray_minus_v0_mean']:+.1f} nats")

    # Three outcomes, not two. A gap that is large but NEGATIVE is not "no improvement" —
    # it is a significant degradation, and collapsing the two would hide a real result.
    gap = out["spray_minus_v0_mean"]
    if gap > noise:
        unpaired = "BEATS v0 (gap exceeds the noise floor)"
    elif gap < -noise:
        unpaired = "is WORSE than v0 (gap exceeds the noise floor, in the wrong direction)"
    else:
        unpaired = "is INDISTINGUISHABLE from v0 (gap within the noise floor)"
    print(f"\n  UNPAIRED verdict: spray at equal capacity {unpaired}")
    out["unpaired_verdict"] = unpaired
    out["unpaired_beats"] = bool(gap > noise)

    if paired_ok:
        sp = np.load(got["spray"]["_lppd_i"])
        for ref in ("v0_a", "v0_b"):
            r = paired_test(sp, np.load(got[ref]["_lppd_i"]))
            lo, hi = r["ci95_nats"]
            r["direction"] = ("better" if lo > 0 else
                              "worse" if hi < 0 else "indistinguishable")
            r["significant"] = bool(lo > 0 or hi < 0)   # CI excludes 0 in EITHER direction
            out[f"paired_vs_{ref}"] = r
            print(f"\n  PAIRED spray - {ref}: {r['total_delta_nats']:+.1f} nats "
                  f"({r['mean_per_event']:+.6f}/event)")
            print(f"    95% CI [{lo:+.1f}, {hi:+.1f}] nats"
                  f"  |  spray better on {r['frac_events_better']:.1%} of events")
            print(f"    -> spray is {r['direction'].upper()}"
                  + (" (CI excludes 0)" if r["significant"] else " (CI includes 0)"))
    else:
        print(f"\n  PAIRED test SKIPPED — holdout order digests differ: {digests}")
        print("    The runs did not score the same events in the same order.")

    print(f"\n  weighted ECE: " + ", ".join(f"{k} {v:.6f}" for k, v in out["ece"].items()))
    print(f"  fit minutes : " + ", ".join(f"{k} {v}" for k, v in out["fit_minutes"].items()))
    print("\n  For reference, at m_trees=50 (Stage 3): spray beat the frozen anchor by "
          f"{M50['spray'] - M50['v0_anchor']:+.1f} nats")
    print(f"  against a {M50['noise_floor']:.1f}-nat noise floor -> judged NO improvement.")
    print("=" * 72)

    outdir = cfg.results_dir / f"capacity_{stage}_m{m}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "capacity_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {outdir}/capacity_metrics.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-trees", type=int, default=200)
    ap.add_argument("--stage", default="C", choices=["A", "B", "C"],
                    help="C is the real experiment. Use A to smoke the whole pipeline "
                         "end to end in ~3 minutes before committing hours to it")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, fit nothing")
    ap.add_argument("--analyze", action="store_true", help="skip fitting, score what exists")
    args = ap.parse_args()
    cfg = load_config()
    m, stage = args.m_trees, args.stage
    plan = fits(m, stage)

    if not args.analyze:
        # Stage C at m_trees=50 took 26.2 min; PGBART cost is ~linear in m_trees.
        est = 26.2 * m / 50 if stage == "C" else 0.4 * m / 50
        print(f"Capacity experiment — stage {stage}, m_trees={m}")
        print(f"  3 fits x ~{est:.1f} min ~= {3 * est / 60:.1f} hours"
              + (", plus ~4 GB of idata.nc per fit (~12 GB total)" if stage == "C" else ""))
        if stage != "C":
            print("  NOTE: stage != C is a wiring smoke only — its numbers are noise.")
        for f in plan:
            state = "DONE" if (cfg.results_dir / f["dir"] / "metrics.json").exists() else "todo"
            print(f"  [{state}] {f['key']:6s} {f['variant']:5s} -> results/{f['dir']}"
                  f"   ({f['what']})")
        if args.dry_run:
            print("\n--dry-run: nothing fitted.")
            return
        print("\nSafe to interrupt: completed fits are skipped on re-run.\n")
        for f in plan:
            if not run_one(cfg, f, m, stage):
                print(f"\nStopping after a failed fit. Re-run to resume from {f['key']}.")
                raise SystemExit(1)

    analyze(cfg, m, stage)


if __name__ == "__main__":
    main()
