# xwobart

Bayesian xwOBA rebuild with credible intervals. v0: BART categorical model over
(launch_speed, launch_angle, sprint_speed) — the public-xwOBA information set —
with full posterior uncertainty and a four-check evaluation harness.

Spec: docs/superpowers/specs/2026-07-17-xwobart-v0-design.md
Plan: docs/superpowers/plans/2026-07-17-xwobart-v0.md
Results: results/RESULTS.md

## Setup

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv
    .venv/bin/pip install -r requirements.txt        # or requirements.lock for exact pins
    cp .env.example .env                             # set STATCAST_PATH

Depends on ../kinferencetoolkit (editable install) for pipeline.statcast_loader
and pipeline.player_names. Pitch data is read from the local monthly Statcast
cache at STATCAST_PATH; the only network pulls are the sprint-speed and
expected-stats leaderboards (cached in data/raw/).

## Run

    .venv/bin/python scripts/run_v0.py --stage A     # 5k-row wiring pass
    .venv/bin/python scripts/run_v0.py --stage B     # 50k development fit + all four checks
    .venv/bin/python scripts/run_v0.py --stage C     # full (config) — needs --acknowledge-runtime if >30 min

    .venv/bin/pytest                                 # unit tests (no MCMC)
    .venv/bin/python scripts/smoke_model.py          # synthetic model smoke

## Key decisions / deviations

See spec §14 and results/RESULTS.md. Highlights: bunt filter uses `des` (not
`description`); out-class linear weight is the empirical ~0.016 (Savant credits
ROE/FC at 0.9); coverage gaps (2022 Oct 1–5, 2023 Oct 1) are reported, not
pulled; per-stage results directories; Stage A predictions capped at 20k rows.
Execution-discovered (see RESULTS.md): OOS prediction uses pymc-bart's
stored-trees predictor (the `set_data` path silently freezes `mu` in 0.12);
`h5netcdf`/`h5py` added for NetCDF idata I/O; BART `mu` R-hat is treated as a
warning (structural, not a convergence stop).
