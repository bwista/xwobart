# xwobart v0 — Bayesian xwOBA rebuild with credible intervals (BART baseline)

Date: 2026-07-17
Status: Approved by user (design discussion), pending implementation
Repo: /Users/jweinga/Documents/python/xwobart (project name: **xwobart**)

## 1. Goal

v0 of a staged model ladder. Reproduce the information set of public Statcast
xwOBA — exit velocity, launch angle, sprint speed — under a BART categorical
model, gaining full posterior uncertainty and a reusable evaluation harness.

v0 is deliberately minimal: **no** added features, batter effects, or park
effects. Its value depends on being a clean like-for-like baseline against the
public metric. Work incrementally: a tiny end-to-end run must work before
anything scales. Ask the user before any step expected to take more than ~30
minutes of compute.

## 2. Non-goals (later rungs)

- v1: spray angle and handedness
- v2: bat tracking (`bat_speed` / `swing_length` exist in the cache from 2024)
- v3: batter partial pooling
- Park effects, defensive positioning, weather — not on the ladder yet
- 2026 partial-season data exists in the cache; unused in v0

## 3. Environment and dependencies

- Python 3.12 virtual environment at `.venv/` (framework build
  `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`; chosen
  for broadest pymc/pytensor wheel support on this machine).
- Pinned `requirements.txt`. Key entries:
  - `-e ../kinferencetoolkit` — editable install of the local KIT package.
    Brings `pymc>=5`, `pytensor`, `arviz`, `polars`, `numpy`, `pyyaml`,
    `python-dotenv`, `pybaseball` as transitive dependencies, plus the
    importable `pipeline` package.
  - Direct: `pymc-bart>=0.5`, `pandas`, `pyarrow`, `matplotlib`, `pytest`.
  - Everything else pinned to the versions resolved at first install.
- KIT is a moving local dependency: `RESULTS.md` records its git SHA at run
  time.
- Environment variable **`STATCAST_PATH`** (KIT's existing convention) points
  at the monthly Statcast parquet directory. Loaded from `.env` via
  python-dotenv. `config.yaml` carries a fallback default
  (`/Users/jweinga/Documents/python/punchoutpredictor/data/statcast`); the env
  var wins when set. `.env` is gitignored; `.env.example` is committed.
- One global seed in `config.yaml`, passed explicitly to every sampler,
  subsampler, and RNG.
- All scripts runnable from the repo root.
- Data transforms in Polars. pandas only at the pybaseball and pymc
  boundaries (convert immediately).
- Machine reality: 11 cores, 18 GB RAM. Memory constraints in §7 are design
  requirements, not suggestions.

## 4. Repo layout

```
xwobart/                     # repo root (this repo)
  config.yaml                # seasons, paths, model settings, subsample sizes, seed
  .env.example               # STATCAST_PATH=<path to monthly statcast parquets>
  requirements.txt
  README.md                  # replaces the current stub; decisions, deviations, runtimes
  src/
    data.py                  # acquisition and caching (via KIT statcast_loader)
    prep.py                  # filtering, outcome classes, feature table
    model.py                 # model build, fit, out-of-sample prediction
    rollup.py                # linear weights, event values, player aggregation
    evaluate.py              # the four acceptance checks
  scripts/
    run_v0.py                # orchestrates everything from config; --stage A|B|C
  tests/                     # pytest unit tests for pure logic
  data/
    raw/                     # slim per-season pitch caches (gitignored)
    processed/               # KIT player_names cache lands here (gitignored)
  results/
    stage_A/  stage_B/  stage_C/   # per-stage: idata.nc, metrics.json, figures/, player_table.parquet
    RESULTS.md               # cross-stage summary
```

`.gitignore` (replaces the current `node_modules/` stub): `.venv/`, `data/`,
`results/**/*.nc`, `__pycache__/`, `.env`, `.pytest_cache/`. Small artifacts
(`metrics.json`, `figures/`, `player_table.parquet`, `RESULTS.md`) stay
tracked.

## 5. Data acquisition (`src/data.py`)

### 5.1 Pitch-level data — local cache only, via KIT

No `pybaseball.statcast()` pulls. Source is the local monthly parquet cache
(`statcast-MM-YYYY.parquet` files) read through
`pipeline.statcast_loader.load_statcast` from kinferencetoolkit:

1. Call `load_statcast(STATCAST_PATH, start_year=2022, end_year=2025)`. Its
   documented warmup behavior also loads `start_year - 1` (2021) files;
   immediately filter to `game_year` in 2022–2025. Its `diagonal_relaxed`
   concat absorbs the cache's mixed `game_date` dtypes (date32 vs string);
   normalize `game_date` to a Date column after load.
2. Filter `game_type == "R"` (the cache contains postseason rows for
   2024–2025).
3. Select only: `game_pk`, `game_date`, `game_year`, `batter`, `events`,
   `description`, `des`, `type`, `bb_type`, `launch_speed`, `launch_angle`,
   `launch_speed_angle`, `estimated_woba_using_speedangle`, `woba_value`,
   `woba_denom`. (`des` and `game_year` are additions to the original column
   list: `des` is required for the bunt filter — see §6 — and `game_year` is
   the season key.)
4. Write one slim parquet per season to `data/raw/statcast-<year>-slim.parquet`.
   Never rebuild if the file exists (`--force-data` flag to override).
   Subsequent runs read only the slim caches.

The one-time build loads ~3.5M rows × 94 columns transiently (a few GB);
acceptable once on this machine.

### 5.2 Coverage validation

`config.yaml` records the expected regular-season window per season:

| Season | Expected window | Cache status at design time |
|--------|-----------------------------|------------------------------|
| 2022 | 2022-04-07 → 2022-10-05 | ends 2022-09-30 — missing Oct 1–5 |
| 2023 | 2023-03-30 → 2023-10-01 | ends 2023-09-30 — missing Oct 1 |
| 2024 | 2024-03-20 → 2024-09-30 | complete |
| 2025 | 2025-03-18 → 2025-09-28 | complete |

After building each season cache, compare its min/max `game_date` to the
window. Gaps are **reported** (stdout + `metrics.json` + `RESULTS.md`), never
pulled. To fill a gap the user runs KIT's own
`pipeline/statcast_loader.py --update --date <YYYY-MM-DD>` (e.g.
`--date 2022-10-05`, `--date 2023-10-01`), then rebuilds with `--force-data`;
`load_statcast` picks up new monthly files automatically.

### 5.3 Sprint speed

`pybaseball.statcast_sprint_speed(year, min_opp=10)` per season 2022–2025,
retry-wrapped, cached to `data/raw/sprint_speed-<year>.parquet`. Merge onto
pitch data by batter id and season. Batters without a qualifying sprint speed
get the league median for that season plus a boolean `imputed_speed` flag.
Report the imputation rate (share of BBE rows imputed, per season).

### 5.4 Public expected stats

`pybaseball.statcast_batter_expected_stats(year)` per season 2022–2025 with
`minPA` low enough to include all 100+ PA players. Cached to
`data/raw/expected_stats-<year>.parquet`. Used only for the player-level
replication check (§9.1).

### 5.5 Player names

`pipeline.player_names` (KIT) resolves batter id → display name for the
player table; it maintains its own cache under `data/processed/` and degrades
silently to raw ids on network failure.

## 6. Data prep (`src/prep.py`)

1. Batted ball events (BBE): rows with `type == "X"`. Exclude bunts by
   filtering rows whose `des` text contains "bunt" (case-insensitive).
   **Deviation from the original prompt:** the pitch-level `description`
   column is always `hit_into_play` for BBE and catches zero in-play bunts;
   `des` catches sac bunts, bunt singles, and bunt outs (verified: 197 vs 0 in
   June 2024).
2. Drop BBE rows with missing `launch_speed` or `launch_angle`; report the
   dropped percentage per season (expected ~0.5%; investigate if materially
   larger).
3. Outcome class `y`: `single`→1, `double`→2, `triple`→3, `home_run`→4,
   everything else (including `field_error`, `fielders_choice`, sac flies)
   →0 (out). Print the class distribution; triples under 1% is expected.
4. Feature matrix `X`: exactly three columns — `launch_speed`,
   `launch_angle`, `sprint_speed`. No standardization (BART does not need
   it). No batter identifiers, ever.
5. Non-BBE plate appearance table for the rollup: rows with
   `woba_denom == 1` and `type != "X"` (strikeouts, walks, HBP, catcher's
   interference), keeping `batter`, season, `woba_value`, `woba_denom`.

## 7. Model (`src/model.py`)

pymc-bart multiclass pattern:

```python
import pymc as pm
import pymc_bart as pmb

k = 5
with pm.Model() as model:
    mu = pmb.BART("mu", X, y, m=cfg.m_trees, shape=(k, n))
    p = pm.Deterministic("p", pm.math.softmax(mu, axis=0))   # Stage A only; see below
    obs = pm.Categorical("y_obs", p=p.T, observed=y)
    idata = pm.sample(tune=cfg.tune, draws=cfg.draws, chains=cfg.chains,
                      random_seed=cfg.seed, compute_convergence_checks=True)
```

### 7.1 Stages

Scale in three explicit stages; **stop and report wall-clock runtime after
each** before proceeding.

| Stage | Rows (stratified subsample) | m_trees | tune | draws | chains |
|-------|------------------------------|---------|------|-------|--------|
| A (wiring) | 5,000 | 20 | 200 | 200 | 2 |
| B (development) | 50,000 | 50 | 500 | 500 | 2 |
| C (full) | all train rows if B extrapolates tolerably, else 100,000 | 50 | 500 | 500 | 2 |

Stage A's only purpose is proving the pipeline runs end to end and produces
every artifact. Stage B runs all four acceptance checks. Stage C's sampler
settings match B (an explicit decision — the original prompt left them
unspecified); the subsample seed and size are recorded in `metrics.json`
either way.

Stratified subsampling preserves class **proportions** (so ~0.6% triples
survive); it never rebalances — the mlb-hit-classifier resampling study
showed every rebalancing scheme distorts calibrated probabilities, and
calibration is an acceptance gate here.

### 7.2 Memory engineering (18 GB machine)

- Storing `mu` for all ~370k train BBE at 2×500 draws ≈ 15 GB in float64 —
  infeasible alongside sampling overhead. Expect Stage C to land on the
  spec-sanctioned 100k fallback unless Stage B extrapolation says otherwise.
- The `p` Deterministic doubles storage; keep it in Stage A (wiring proof),
  drop it from the graph in Stages B/C and recompute softmax from `mu` on
  demand.
- Save `InferenceData` to `results/stage_<X>/idata.nc`.

### 7.3 Out-of-sample prediction (2025 holdout)

Follow the current pymc-bart documented approach — `pm.Data` container for
`X`, `pm.set_data` with holdout features, `pm.sample_posterior_predictive` —
predicting from the stored trees without refitting. Verify the mechanism at
Stage A by predicting the training rows and confirming agreement with
in-sample `p`. If the static `shape=(k, n)` blocks `set_data`, the fallback
is direct prediction from the fitted trees (pymc-bart's stored-trees
utilities), validated the same way. Predict in chunks of ~20k events with
draws thinned to ≤500 total.

### 7.4 Failure handling

If sampling diverges, `R-hat`/ESS checks fail, or class probabilities
collapse, stop and report — do not paper over it.

## 8. Linear weights and rollup (`src/rollup.py`)

1. Linear weights `w_k` per class: mean observed `woba_value` by outcome
   class over the **training** seasons' BBE. Sanity magnitudes from June 2024:
   single 0.900, double 1.250, triple 1.600, home run 2.000. **The out class
   is ~0.016, not 0**: Savant credits `field_error` and `fielders_choice`
   rows with `woba_value` 0.9, and those events map to the out class. The
   spec formula (empirical mean per class) wins over the original prompt's
   "out will be 0" parenthetical; actual `w_k` values go in `metrics.json`.
2. Expected event value: for each posterior draw, dot the class
   probabilities with `w`. Keep the draw dimension. Thin to ≤500 draws and
   process events in chunks (float32 is acceptable for the value matrix).
3. Player-season rollup, matching the public xwOBA construction, computed
   **per posterior draw**:
   - numerator = Σ expected event values over the player's BBE
     + Σ actual `woba_value` over the player's non-BBE PA rows
   - denominator = Σ `woba_denom` over both
4. Output `results/stage_<X>/player_table.parquet`: `batter`, `player_name`,
   `season`, `PA` (= Σ `woba_denom`), posterior mean, sd, 5th and 95th
   percentiles, and public Savant xwOBA (`est_woba`) joined on for
   comparison. Train seasons use in-sample predictions; the holdout season
   uses out-of-sample predictions.

## 9. Acceptance checks (`src/evaluate.py`)

The four v0 gates. Each produces a figure or table under
`results/stage_<X>/figures/` and writes its numbers to `metrics.json`.

### 9.1 Replication

- Event-level Pearson r between posterior-mean expected value and
  `estimated_woba_using_speedangle` on BBE, reported separately for train
  and holdout. Scatter plot each.
- Player-season Pearson r between rollup posterior mean and public xwOBA for
  players with 100+ PA, reported for train seasons and holdout separately.
  Scatter plot.
- Mean residual against binned EV and binned LA (not just the single
  correlation) to expose structure.

### 9.2 Calibration

On the 2025 holdout: one reliability curve per outcome class using 10
quantile bins of predicted probability; per-class Brier score; expected
calibration error. Near-empty bins (triples) are handled gracefully: collapse
duplicate quantile edges, annotate bin counts, never divide by zero.

### 9.3 Sprint speed localization

- Predicted value as a function of sprint speed (grid ~23–31 ft/s) at two
  fixed contact points via the OOS prediction path: topped grounder
  (`launch_speed` 85, `launch_angle` −10) and barrel (103, 28). Expected
  shape: grounder curve slopes up, barrel curve flat.
- Using `launch_speed_angle` codes, compare the model's sprint-speed effect
  (slope/correlation of predicted value vs sprint speed) inside weak/topped
  contact (codes 1–2) against solid/barreled contact (codes 5–6).
- Report pymc-bart variable importance for the three features.

### 9.4 ELPD anchor and undercorrection test

- Pointwise log score on the 2025 holdout. Primary anchor: standard lppd,
  `Σ_i log( mean_draws p(y_i | θ_s) )`, with its standard error
  (`sqrt(n · var(pointwise))`), stored in `metrics.json` as the fixed anchor
  v1+ must beat. Also store the mean-of-log variant (the original prompt's
  literal phrasing) for transparency; both labeled clearly.
- Undercorrection test: on holdout ground balls (`bb_type == "ground_ball"`),
  correlate `(actual woba_value − model predicted value)` with sprint speed;
  side by side, correlate `(actual woba_value − estimated_woba_using_speedangle)`
  with sprint speed. Public xwOBA is known to undercorrect for speed; the
  model should shrink that correlation toward zero.

## 10. Orchestration (`scripts/run_v0.py`)

`python scripts/run_v0.py --stage A|B|C [--force-data]` runs, from config:
data build (idempotent) → prep → fit → OOS prediction → rollup → evaluate →
report. Per-stage outputs under `results/stage_<X>/` so stages never clobber
each other (deviation from the original flat `results/` — required because
stage artifacts must coexist). Runtime is reported after fit and after the
full stage; the >30-minute ask-first rule applies to Stage C in particular.

## 11. Reporting (`results/RESULTS.md`)

Factual, no adjectives: data volumes and drops per season, coverage gaps,
sprint-speed imputation rates, KIT git SHA, stage runtimes, the four checks'
headline numbers per stage, linear weights used, subsample seed/size, and any
deviations from this spec with reasons.

## 12. Testing

Pytest unit tests for pure logic only — no MCMC in tests (Stage A is the
integration test):

- outcome class mapping (including `field_error`/`fielders_choice` → out)
- bunt exclusion via `des`
- missing-LS/LA drop accounting
- stratified subsampler preserves proportions and the seed reproduces
- linear-weights computation on a toy frame
- per-draw rollup arithmetic on a tiny fixture (hand-checkable numerator /
  denominator)
- coverage validator (gap detection against a season window)
- non-BBE PA table filter

## 13. Guardrails

- No features beyond the three named. No batter identifiers in `X`.
- No silent changes to seasons, thresholds, or sampler settings — changes go
  in `config.yaml` and `RESULTS.md`.
- pybaseball pulls (sprint speed, expected stats only) are chunked, cached,
  retried, never hammered.
- Divergences or collapsed class probabilities: stop and report.
- Ask before any step expected to exceed ~30 minutes of compute.

## 14. Deviations from the original prompt-starter

| Change | Reason |
|--------|--------|
| Pitch data from local cache via KIT `load_statcast`; no `statcast()` pulls | User decision; cache verified complete for 2022–2025 except noted tail gaps |
| Env var named `STATCAST_PATH` (not invented anew) | Matches KIT's existing `.env` convention |
| Editable dependency on kinferencetoolkit | User request: import its modules rather than rewrite |
| Bunt filter on `des` instead of `description` | `description` catches zero in-play bunts (verified) |
| `game_type == "R"` filter + `des`, `game_year` columns added | Cache contains postseason rows; bunt filter and season key need the extra columns |
| Out-class linear weight empirical (~0.016), not forced 0 | Savant credits ROE/FC at 0.9; the spec's own formula produces this |
| Per-stage `results/stage_<X>/` subdirectories | Stage artifacts must not clobber each other |
| `p` Deterministic dropped from Stages B/C | Memory: it doubles idata size for no information gain |
| Stage C sampler settings pinned to B's | Original prompt left them unspecified; explicit beats silent |
| ELPD primary = lppd (log-of-mean); mean-of-log also stored | lppd is the standard ELPD estimator; both kept for transparency |
| Repo/project named `xwobart` at repo root | User decision |
| Coverage gaps documented, not pulled (2022 Oct 1–5, 2023 Oct 1) | User chose local-cache-only; fill path via KIT documented in §5.2 |

## 15. Definition of done (v0)

1. Stage A produces every artifact end to end (idata, metrics, figures,
   player table, RESULTS entry) on 5k rows.
2. Stage B completes with runtime reported and all four acceptance checks
   computed with plausible values.
3. Stage C decision (full vs 100k fallback) made from B's extrapolation,
   recorded, and executed after user sign-off on runtime.
4. `results/RESULTS.md` complete per §11.
5. All unit tests green.
