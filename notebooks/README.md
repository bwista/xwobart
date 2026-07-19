# xwobart review notebooks

Interactive walk-through of everything accomplished so far. Each notebook **loads the
already-computed artifacts** in `results/` and displays the saved figures — nothing here
re-fits a model or recomputes heavy results, so they run in seconds.

## How to run

Jupyter is **not** in the project `.venv` yet. Install it, then launch with the venv kernel:

```bash
cd /Users/jweinga/Documents/python/xwobart
.venv/bin/python -m pip install jupyterlab      # or: notebook / ipykernel for VS Code / Cursor
.venv/bin/python -m jupyterlab notebooks/       # opens in the browser
```

In VS Code / Cursor, just open a notebook and pick the `.venv` interpreter as the kernel.
Run the first cell (setup) first — it finds the repo root automatically, so it works whether
you launch from the repo root or from `notebooks/`.

## The notebooks (read in order)

| # | Notebook | What it shows | Reads |
|---|----------|---------------|-------|
| 01 | `01_v0_model_quality.ipynb` | The v0 BART model reproduces Savant xwOBA at player-r ≈ 0.96; calibration, sprint signal, feature importance; quality saturates by ~50k rows | `results/stage_{A,B,C}/` |
| 02 | `02_accuracy_vs_savant.ipynb` | Is v0 *more accurate* than Savant at predicting next-season wOBA? → **statistical parity** (r 0.481 vs 0.487), both beat naive | `results/benchmark/` |
| 03 | `03_uncertainty_bands.ipynb` | v0's model interval is **flat in PA** (wrong object); a **bootstrap over a player's PAs** narrows correctly with sample size | `results/task_a/`, `results/player_ci/` |
| 04 | `04_talent_estimates.ipynb` | **Phase 1 empirical-Bayes true-talent xwOBA** — shrinkage, reliability, examples, and the next-season validation (beats raw; ties Savant, beats it at low PA) | `results/talent/` |

## The arc in one paragraph

v0 is a faithful, well-calibrated xwOBA model (01) — but it's at Savant's accuracy ceiling
with three features (02). The project then pivoted from accuracy to **uncertainty**: v0's
posterior band doesn't shrink with sample size, but a PA-bootstrap does (03). Phase 1 puts
those together into a **sample-size-regressed true-talent estimate** with a calibrated
interval, which beats the raw number and edges Savant for low-PA hitters (04). Next is
**Phase 2** — replacing the flat league-mean prior with a BART contact-quality prior
(`docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md`).

For the written versions of these findings, see `results/RESULTS.md` and each
`results/*/NOTES.md`.
