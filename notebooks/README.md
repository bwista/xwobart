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

## The notebooks (read in order)

| # | Notebook | What it shows | Reads |
|---|----------|---------------|-------|
| 01 | `01_v0_model_quality.ipynb` | The v0 BART model reproduces Savant xwOBA at player-r ≈ 0.96; calibration, sprint signal, feature importance; quality saturates by ~50k rows | `results/stage_{A,B,C}/` |
| 02 | `02_accuracy_vs_savant.ipynb` | Is v0 *more accurate* than Savant at predicting next-season wOBA? → **statistical parity** (r 0.481 vs 0.487), both beat naive | `results/benchmark/` |
| 03 | `03_uncertainty_bands.ipynb` | v0's model interval is **flat in PA** (wrong object); a **bootstrap over a player's PAs** narrows correctly with sample size | `results/task_a/`, `results/player_ci/` |
| 04 | `04_talent_estimates.ipynb` | **Phase 1 empirical-Bayes true-talent xwOBA** — shrinkage, reliability, examples, and the next-season validation (beats raw; ties Savant, beats it at low PA) | `results/talent/` |
| 05 | `05_level2_talent.ipynb` | **Phase 2 / Stage 1** — shrink toward what the *contact* implies (joint MVN over xwOBA + exit velo + barrel rate) instead of the league mean; gain concentrated at low PA (+0.072 at 30–60), the shared-noise tripwire, and why the win isn't statistically established | `results/talent2/` |

## The arc in one paragraph

v0 is a faithful, well-calibrated xwOBA model (01) — but it's at Savant's accuracy ceiling
with three features (02). The project then pivoted from accuracy to **uncertainty**: v0's
posterior band doesn't shrink with sample size, but a PA-bootstrap does (03). Phase 1 puts
those together into a **sample-size-regressed true-talent estimate** with a calibrated
interval, which beats the raw number and edges Savant for low-PA hitters (04). Phase 2 then
fixes the last weakness — that everyone regresses toward the *league* mean — by putting the
fast-stabilizing peripherals (exit velo, barrel rate) into the prior, so a rookie barreling
the ball regresses toward a slugger (05). That help lands exactly where the sample is thin
(+0.072 r at 30–60 PA, nothing at 250+) and the tripwire for the obvious failure mode came
back clean — though the pooled effect is small and one holdout season disagrees.

Note the design that *didn't* survive: shrinking toward the BART model's own xwOBA was a
structural no-op (the surface is built from the same batted balls as the raw number), so the
old roadmap `docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md` is superseded by
`docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md`.

For the written versions of these findings, see `results/RESULTS.md` and each
`results/*/NOTES.md`.
