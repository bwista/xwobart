# Handoff — xwobart forecaster: open levers after the spray investigation

Paste the block below into a fresh conversation at the repo root
(`/Users/jweinga/Documents/python/xwobart`).

---

Continue the xwobart rest-of-season forecaster. The **spray** thread is closed; three levers remain
open. Everything referenced below is on `origin/main` (@ `9df168c`), tests green (89). Read the
pointers, pick a lever (I recommend the first), and use the project's **brainstorm → spec → plan →
subagent-driven TDD** workflow on a feature branch (not `main`).

## Where things stand (as of 2026-07-23)

xwobart rebuilds Bayesian xwOBA and pushes past the public number. Shipped + validated:
- **v0 BART surface** — replicates Savant (player r 0.96). `src/model.py`, `scripts/run_v0.py`.
- **Talent** — Phase 1 EB (`src/talent.py`), Level 2 joint-MVN over (xwOBA, avg EV, barrel)
  (`src/talent2.py`).
- **Rest-of-season forecast, rung a** (`src/talent3.py`, `scripts/run_talent3.py`, spec
  `docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-forecast-design.md`) — stand at a
  mid-season cutpoint (first *k* PAs) and forecast a hitter's final-season xwOBA. Career random
  intercept + iid drift, xwOBA-only. Beats naive / Marcel / single-season L2 on final-line RMSE
  (G2 CI excludes 0). **Its one open weak spot: G4 calibration — the 50/80% central intervals run
  ~5–7pp narrow at short runway (worst at high *k*).** Full writeup: `results/talent3/NOTES.md`.

**The product yardstick is IN-SEASON / rest-of-season forecasting — NOT next-season.** Score against
rest-of-season realized xwOBA, not next-season RMSE. (Much of the repo's *validation* history is
next-season; treat that as secondary.)

### Spray is RESOLVED — do not reopen it
Two coherent halves, both committed:
1. **Description — spray WINS.** The m=200 capacity experiment (`scripts/capacity_experiment.py`,
   `results/capacity_C_m200/capacity_metrics.json`, `RESULTS.md` §"Spray at matched capacity")
   showed the 5-feature spray BART surface beats v0 by **+3,017 nats** paired (CI excludes 0) vs a
   **37-nat** noise floor — reversing the Phase-2 Stage-3 m=50 negative as a capacity artifact.
2. **Forecast — spray is REDUNDANT.** A cheap pre-check (`src/precheck.py`,
   `python scripts/run_talent3.py --precheck`, `results/talent3/NOTES.md` §"Rung b — spray
   peripheral") showed a hitter's pull tendency adds **ΔR² ≤ +0.0022** for rest-of-season xwOBA
   beyond early xwOBA/EV/barrel (0.01 go bar; well-powered; season-FE robust).

**Conclusion: spray improves *describing* a batted ball but not *forecasting* a hitter's future — it
is a dead forecasting lever. Do NOT add pull to the forecaster.**

### Reusable machinery already built (spray branch, merged to main)
- `build_pa_frame` (`src/talent3.py`) carries per-PA `ev`, `barrel`, `pull`.
- `bootstrap_S` (`src/talent2.py`) takes an optional 4th channel; the 3-channel path is bit-identical.
- `src/precheck.py` — a general incremental-signal go/no-go gate (`pull_incremental_signal`).
- **The rung-b measurement math is specced and double-reviewed** in
  `docs/superpowers/specs/2026-07-23-xwobart-forecast-rungb-spray-peripheral-design.md` §4.2: a
  conditional-prior "flat-θ marginal likelihood" that borrows through the *talent* covariance `Σ_θp`
  and shrinks exactly once (unbiased; reduces to rung a when `Σ_θp=0`). The algebra was independently
  re-derived and confirmed. **Reuse this construction for Lever 1 verbatim — just drop `pull` from the
  channel set.** The implementation plan (`docs/superpowers/plans/2026-07-23-forecast-rungb-spray.md`,
  Tasks 4–8) is the build recipe; only the pre-check (Task 3) STOPped the spray version.

## The open levers — pick one

### 1. Rung b — EV/barrel peripheral measurement  (RECOMMENDED: most shovel-ready)
**Question:** does folding EV + barrel rate (NOT pull) into the within-season talent *measurement*
sharpen the low-PA forecast beyond xwOBA-only rung a? The pre-check controlled *for* EV/barrel, so
whether they help *as measurement peripherals over an xwOBA-only read* is genuinely untested. This is
the original rung b of the parent spec (§4.2/§6), and nearly everything for it already exists.
- **Do the cheap pre-check first:** extend `src/precheck.py` to ask whether early-*k* EV/barrel
  predict rest-of-season xwOBA beyond early-*k* xwOBA alone. Go/no-go before building.
- **If GO, build** with `channels = ("xwoba", "avg_ev", "barrel_rate")` using the validated §4.2
  `peripheral_measurement`, feeding the unchanged rung-a hierarchy. Follow plan Tasks 4–8.
- **Guardrails:** `precompute_full_measurements` must stay **channel-gated** so `channels=("xwoba",)`
  routes through the scalar path and reproduces rung-a numbers **bit-identically** (plan Task 5/6/7
  notes; the k-channel `np.cov` differs ~1 ULP, so use `allclose`, not `array_equal`, off that path).
  Score on the existing G1–G5 panel + a spray-free ablation.

### 2. G4 calibration fix — the short-runway coverage miss
Rung a's 50/80% intervals are narrow, worst when little season remains. `results/talent3/NOTES.md`
limitation 8 diagnoses it: the forward-bootstrap's future-sampling term shrinks with `m` while
whatever the model is missing does not. Levers: recalibration / variance handling at small `m`, or
the surface-uncertainty term the spec deliberately shelved. **Overlaps Lever 1** — peripheral
sharpening may itself tighten low-PA coverage, so consider doing 1 first and re-measuring G4.

### 3. Rung c — aging + AR(1) drift  (blocked on external data)
A shared aging curve `g(age; β)` and `u_{i,t} = ρ·u_{i,t−1} + e` so last season informs this one
beyond the career mean. **Needs player birthdates** — not in the slim Statcast cache; requires a
Chadwick/KIT register join (same metadata path as player-name / sprint resolution). Second-order
gains on top of (a)+(b). Acquire birthdates first, or defer.

## Conventions to follow
- **A cheap pre-check before an expensive build** — the pattern that saved 7 tasks on the spray
  rung. Always extend/reuse `src/precheck.py` to measure incremental signal before committing to a
  model.
- **Reduction-identity + leakage tests are load-bearing.** Any measurement rung must reduce to rung a
  exactly via the scalar `channels=("xwoba",)` route, and `assert_causal` (first-*k* PAs only, no
  future, no later season) must stay green on every forecast.
- **Workflow (rigid):** brainstorm (+ spec-document-reviewer) → writing-plans (+ plan-reviewer) →
  subagent-driven-development (implementer + spec-compliance review + code-quality review per task,
  fix loops until both pass). Commit per task; feature branch.
- **Gotchas:** polars needs explicit `.sort()` for reproducible RNG streams; talent3 is closed-form
  (no BART — the pymc-bart non-reproducibility gotcha doesn't apply here); `bootstrap_S`'s k-channel
  `np.cov` is ~1 ULP unstable across row counts.

## Reproduce the current state
```bash
.venv/bin/python -m pytest -q                          # 89 passed
.venv/bin/python scripts/run_talent3.py --precheck     # the spray STOP verdict
.venv/bin/python scripts/run_talent3.py                # rung-a forecast sweep + gate panel
```
