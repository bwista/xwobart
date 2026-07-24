# xwobart

Bayesian xwOBA, rebuilt from scratch and then pushed past the public number: a BART
outcome surface with full posterior uncertainty, sample-size-honest player intervals,
empirical-Bayes true-talent estimates, and a rest-of-season forecast — every layer
validated against next-season **actual** wOBA, with the failures written up alongside
the wins.

One-line status: the surface **replicates** Savant (player r ≈ 0.96) and sits at its
information ceiling (statistical parity on prediction), so the value added since then
is in *uncertainty* and *talent* — regressed talent beats Savant on low-PA seasons,
and the rest-of-season forecast beats Marcel with a CI that excludes zero.

## The story so far

1. **v0 surface.** BART categorical model over (launch_speed, launch_angle,
   sprint_speed) — the public-xwOBA information set. Replicates Savant at player
   r 0.956, weighted ECE 0.042; holdout ELPD **−80,107 ± 244** over 122k events is the
   frozen anchor any surface change must beat. Quality saturates by ~50k training rows.
2. **Benchmark.** Does v0 predict next-season actual wOBA better than Savant? No —
   pooled r 0.481 vs 0.487, bootstrap CI straddles zero → **statistical parity**. Three
   features are at Savant's ceiling, so the project pivoted from accuracy to
   uncertainty and talent. (`results/benchmark/`)
3. **Uncertainty.** v0's posterior band is flat in PA — it measures *surface*
   uncertainty, not sample size. A bootstrap over each player's own PAs gives the
   honest band; the two cross near 400 PA, below which the model band is up to ~1.8×
   too narrow. (`results/task_a/`, `results/player_ci/`)
4. **Talent, Phase 1.** Gaussian–Gaussian empirical Bayes regresses each raw xwOBA
   toward the per-season league mean by reliability τ²/(τ²+SE²). Beats raw everywhere;
   beats Savant once low-PA seasons are admitted (0.467 vs 0.452). (`results/talent/`)
5. **Talent, Level 2.** Joint MVN treats (xwOBA, avg EV, barrel rate) as noisy
   measurements of correlated talents, so a rookie crushing the ball regresses toward
   a slugger instead of the league mean. The gain lands exactly where designed —
   **+0.072 r at 30–60 PA**, zero at 250+ — while the pooled effect is small and one
   holdout season disagrees; a shared-noise tripwire proves the gain is not a
   same-sample artifact. (`results/talent2/`)
6. **Spray surface — description-yes, prediction-no.** Caches rebuilt with hit
   coordinates, `spray_pull` + `stand_R` added. At the frozen m=50 budget the 5-feature
   refit looked like a negative — the apparent +231-nat gain sat inside the *measured*
   267-nat run-to-run noise floor. That was **capacity dilution**, now confirmed: at a
   matched m=200 budget (`scripts/capacity_experiment.py`, ~4.7 h) spray **decisively
   beats** v0 for *description* — **+3,017 nats** paired against a **37-nat** noise floor,
   82× the floor — so where the ball is hit genuinely improves how the model values a
   batted ball. But it does **not** breach the prediction wall (spray-conditioned rollups
   still lose to v0), and calibration regresses (ECE 0.0369 vs 0.0277).
   (`results/capacity_C_m200/`, `results/stage_C_{m200a,m200b,spray_m200}/`,
   `results/stage2_rebuild/`, `results/stage_C_spray/`, `results/rollup_ab/`)
7. **Rest-of-season forecast (talent3, rung a).** Stand at a hitter's first *k* PAs and
   forecast his final-season xwOBA, adding a career random intercept over Phase 1.
   Pooled RMSE **0.0220** beats Marcel (0.0227) and single-season Level 2 (0.0245),
   both with paired-bootstrap CIs excluding zero; the 50%/80% interval coverage runs
   narrow — a real, open failure (G4). Spray is a **dead lever here**: a hitter's pull
   tendency is forecast-redundant (ΔR² ≤ +0.0022) once early EV/barrel are in.
   (`results/talent3/`)

`results/RESULTS.md` is the full ledger — every stage's metrics, gate outcomes,
deviations, and the sampler-reproducibility measurement that bounds all surface
comparisons. Each `results/*/NOTES.md` is the deep-dive for its layer.

## Where the model actually lives

**Code.**

| piece | where |
|---|---|
| BART surface: build/fit, sanity checks, stored-trees OOS prediction | `src/model.py` |
| BBE filtering, outcome classes, features (v0 and spray variants) | `src/prep.py` |
| linear weights, per-draw expected values, player-season rollup | `src/rollup.py` |
| the four acceptance checks (replication, calibration, ELPD, localization) | `src/evaluate.py` |
| data acquisition + caching (Statcast slim caches, leaderboards) | `src/data.py` |
| talent layers: Phase 1 EB / Level 2 MVN / rest-of-season | `src/talent.py`, `src/talent2.py`, `src/talent3.py` (+ `src/forecast.py`, `src/benchmarks.py`) |
| stage sizes, sampler settings, seasons, paths | `config.yaml` |

**Fitted artifacts.** The heavy outputs of a fit — `idata.nc` traces, per-event draw
arrays (`ev_draws_*.npy`), pickled trees — are **gitignored** (hundreds of MB to GB)
and exist only on the machine that ran the fit. What git carries per run directory
under `results/` is the small, sufficient summary: `metrics.json`,
`player_table.parquet`, figures, `NOTES.md`. A fresh clone can therefore read every
result and run every notebook, but **cannot re-score the surface without refitting**
(v0's trees are not persisted; the spray run pickles its trees, also gitignored).

**No-refit layers.** Everything downstream of the surface — the benchmark, the player
bands, and all three talent layers — deliberately runs off public Savant per-event
values from the slim Statcast caches. Closed-form, seconds to run, and provably
unmoved by a BART refit.

## The notebooks

`notebooks/01–03` are the readable walk-through, committed **with outputs embedded** so
everything is viewable on GitHub without running anything. They are organised by **finding**,
not by the chronology of the work: **01 — the surface and its ceiling** (v0's Savant-parity
wall, and spray winning description at matched capacity but not prediction), **02 — uncertainty
and true talent** (the honest sample-size band, empirical-Bayes talent, the peripheral-informed
prior), and **03 — the product** (the rest-of-season forecast, and the spray dead-end for
forecasting). Each ends with a guard cell that asserts the prose's numbers against the
artifacts. See `notebooks/README.md` for the reading order and the per-part arc.

## Setup

    python -m venv .venv
    # activate it: source .venv/bin/activate      (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt               # or requirements.lock for exact pins
    cp .env.example .env                          # set STATCAST_PATH

Depends on `../kinferencetoolkit` (editable install) for `pipeline.statcast_loader`
and `pipeline.player_names`. Pitch data is read from the local monthly Statcast cache
at `STATCAST_PATH`; the only network pulls are the sprint-speed and expected-stats
leaderboards (cached in `data/raw/`). `data/` is gitignored — refitting on a new
machine means pointing `STATCAST_PATH` at a cache first.

## Refit / reproduce

Run everything from the repo root. The surface (the only expensive part):

    python scripts/run_v0.py --stage A     # 5k-row wiring pass          (~1 min)
    python scripts/run_v0.py --stage B     # 50k development fit          (~17 min)
    python scripts/run_v0.py --stage C     # 100k config default          (~30 min; --acknowledge-runtime if over)
    python scripts/run_v0.py --stage C --variant spray   # 5-feature spray surface (needs rebuilt caches)

Each stage writes `results/stage_<X>/` and rewrites its section of
`results/RESULTS.md` between the `<!-- stage_X -->` markers. Downstream layers
(no BART refit; seconds to a few minutes each):

    python scripts/benchmark_vs_savant.py       # v0 vs Savant next-season race   -> results/benchmark/
    python scripts/task_a_uncertainty.py        # model-interval sanity           -> results/task_a/
    python scripts/player_ci_bootstrap.py       # sample-size-aware player bands  -> results/player_ci/
    python scripts/run_talent.py                # Phase 1 EB talent               -> results/talent/
    python scripts/run_talent2.py --stage full  # Level 2 (peripheral prior)      -> results/talent2/
    python scripts/run_talent3.py               # rest-of-season forecast         -> results/talent3/

The spray phase, in order: `rebuild_caches.py` → `qc_spray.py` →
`run_v0.py --variant spray` → `marginalize_spray.py` → `rollup_ab.py`.

Checks:

    pytest                                 # unit tests (no MCMC)
    python scripts/smoke_model.py          # synthetic model smoke (~1-3 min)

Design docs live under `docs/superpowers/` — specs and plans per phase (v0
2026-07-17, phase 2 2026-07-19, rest-of-season forecast 2026-07-20).

## Key decisions / deviations

See spec §14 and `results/RESULTS.md`. Highlights: bunt filter uses `des` (not
`description`); out-class linear weight is the empirical ~0.016 (Savant credits
ROE/FC at 0.9); coverage gaps (2022 Oct 1–5, 2023 Oct 1) are reported, not
pulled; per-stage results directories; Stage A predictions capped at 20k rows.
Execution-discovered (see RESULTS.md): OOS prediction uses pymc-bart's
stored-trees predictor (the `set_data` path silently freezes `mu` in 0.12);
`h5netcdf`/`h5py` added for NetCDF idata I/O; BART `mu` R-hat is treated as a
warning (structural, not a convergence stop); pymc-bart fits are **not
reproducible across processes** even with a fixed seed — the measured ~267-nat
run-to-run ELPD spread is the noise floor every surface comparison is judged
against.
