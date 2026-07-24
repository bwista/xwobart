# xwobart review notebooks

Interactive walk-through of everything accomplished so far. Each notebook **loads the
already-computed artifacts** in `results/` and displays the saved figures — nothing here
re-fits a model or recomputes heavy results, so they run in seconds.

The notebooks are committed **with their outputs embedded**, so every table and figure is
viewable on GitHub or in any notebook viewer straight from a clone — no kernel, no re-run,
no model artifacts needed. Re-running is only for regenerating those outputs.

## How to run

Any Python environment with a Jupyter kernel and `polars` works (the heavy model traces are
not needed — everything reads the committed JSON/parquet artifacts in `results/`):

```bash
python -m pip install jupyterlab polars   # or just ipykernel for VS Code / Cursor
python -m jupyterlab notebooks/           # opens in the browser
```

Run the first cell (setup) first — it finds the repo root automatically and imports the
shared helpers from `notebooks/nb_helpers.py` (`jload`, `show_fig`, the polars display
defaults), so it works whether you launch from the repo root or from `notebooks/`.

Each notebook ends with a small **guard cell** asserting that the headline numbers quoted
in the prose still match the artifacts under `results/` — if a pipeline re-run changes a
result, re-executing the notebook fails loudly at the guard instead of letting the text
silently drift.

## The three parts (read in order)

The series is organised by **finding**, not by the chronology of the work — each part opens
by answering the question the previous one raised and closes by raising the next.

| # | Notebook | The finding it delivers | Reads |
|---|----------|-------------------------|-------|
| 01 | `01_surface_and_ceiling.ipynb` | v0 replicates Savant (player r **0.956**) but sits at its **information ceiling for prediction** (next-season parity, r 0.481 vs 0.487). Spray — the information Savant lacks — genuinely improves *description* at adequate capacity (m=200: **+3,017 nats** paired vs a **37-nat** noise floor, reversing the m=50 negative as capacity dilution), yet does **not** breach the prediction wall (spray rollups still lose to v0) and calibration regresses under it. | `results/stage_{A,B,C}/`, `results/benchmark/`, `results/stage2_rebuild/`, `results/stage_C_spray/`, `results/capacity_C_m200/`, `results/stage_C_{m200a,spray_m200}/`, `results/rollup_ab/` |
| 02 | `02_uncertainty_and_talent.ipynb` | v0's posterior band is the **wrong object** (flat in PA); a PA-bootstrap narrows correctly and the two cross near 450–600 PA. Empirical-Bayes shrinkage turns that into a true-talent estimate that beats raw everywhere and edges Savant once low-PA seasons are admitted (**0.467 vs 0.452**). Level 2 puts the fast-stabilizing peripherals in the prior: **+0.072 r at 30–60 PA**, ~nothing at 250+, pooled effect small and one holdout season disagrees — with the shared-noise tripwire coming back clean. | `results/task_a/`, `results/player_ci/`, `results/talent/`, `results/talent2/` |
| 03 | `03_forecast.ipynb` | From a hitter's first *k* PAs, forecast his final-season xwOBA with a range. Pooled RMSE **0.0220** beats naive / Marcel / single-season Level 2 (bootstrap CIs exclude 0); G5 reduces to Phase 1 exactly (**5.6e-17**); **G4 calibration fails** — 50/80% intervals run narrow, worst at short runway. And **spray adds nothing here**: pull tendency is forecast-redundant (**ΔR² ≤ +0.0022** beyond early xwOBA/EV/barrel). | `results/talent3/` |

## The arc in three paragraphs

**01 — The surface and its ceiling.** v0 is a faithful, well-calibrated reconstruction of
xwOBA from three contact features (player r 0.956, weighted ECE 0.042), but at those three
features it sits *exactly* at public Savant's ceiling for predicting next-season actual wOBA
(r 0.481 vs 0.487, parity). The one honest way past that ceiling is to feed the surface what
Savant throws away — where the ball was hit, and which hand hit it. Doing so is a real *description*
win, but only once the tree budget is large enough to use it: at m=50 spray looked like a
negative (inside a 267-nat run-to-run noise floor), and it took re-measuring that noise floor at
matched m=200 capacity to reveal a **+3,017-nat** improvement against a 37-nat floor. That better
surface still doesn't predict a hitter's next season any better, and its calibration is worse —
description-yes, prediction-no.

**02 — Uncertainty and true talent.** If the point estimate is capped, the value is in the *band*
and the *center*. v0's posterior band is flat in PA — it measures surface uncertainty, not sample
size — while a bootstrap over a player's own PAs gives an honest band that narrows correctly, the
two crossing near 450–600 PA. Empirical-Bayes shrinkage turns that into a sample-size-honest
true-talent estimate that beats the raw number everywhere and edges Savant for low-PA hitters
(0.467 vs 0.452). Level 2 then replaces the flat league-mean prior with what a hitter's *contact
quality* implies, so a rookie barreling the ball regresses toward a slugger; the help lands exactly
where the sample is thin (+0.072 r at 30–60 PA, nothing at 250+), the pooled effect is small, one
holdout season disagrees, and the tripwire for the obvious artifact came back clean and decisive.

**03 — The product: forecasting the rest of a season.** The first two parts describe seasons that
already happened; this one forecasts one in progress. Standing at a hitter's first *k* PAs it
forecasts his final full-season xwOBA with a calibrated range, adding a career random intercept
over his prior seasons — and beats naive, Marcel and single-season Level 2 with bootstrap CIs that
exclude zero (pooled RMSE 0.0220). G5 reduces to Phase 1 exactly; G4 is an honest, open failure —
the 50/80% intervals run narrow, worst when little season is left. And a well-powered pre-check
closes the spray thread for good: a hitter's pull tendency is forecast-redundant once his early
exit velocity and barrel rate are in the model.

Note the design that *didn't* survive: shrinking toward the BART model's own xwOBA was a
structural no-op (the surface is built from the same batted balls as the raw number), so the
old roadmap `docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md` is superseded by
`docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md`.

For the written versions of these findings, see `results/RESULTS.md` and each
`results/*/NOTES.md`.
