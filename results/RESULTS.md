# xwobart v0 results

<!-- stage_A -->
## Stage A
- kit_sha: 648b990 | seed: 42 | fit rows: 5000 | predict rows: {'train': 20000, 'holdout': 20000, 'capped': True}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 0.4 min | total: 0.6 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.845, event holdout 0.840, player train 0.453, player holdout 0.455
- calibration — weighted ECE 0.0801
- ELPD (lppd) -14401.3 ± 100.6 over 20000 events
- undercorrection corr — model 0.028 vs public -0.002
- localization slopes (per ft/s) — grounder 0.0010, barrel -0.0039
- sanity warnings: ['max R-hat on probed mu cells = 1.828 (> 1.1)']
<!-- /stage_A -->

<!-- stage_B -->
## Stage B
- kit_sha: 648b990 | seed: 42 | fit rows: 50000 | predict rows: {'train': 363595, 'holdout': 122006, 'capped': False}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 14.2 min | total: 17.1 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.915, event holdout 0.916, player train 0.960, player holdout 0.963
- calibration — weighted ECE 0.0376
- ELPD (lppd) -80023.5 ± 246.5 over 122006 events
- undercorrection corr — model 0.033 vs public 0.013
- localization slopes (per ft/s) — grounder 0.0023, barrel -0.0034
- sanity warnings: ['max R-hat on probed mu cells = 1.666 (> 1.1)']
<!-- /stage_B -->

<!-- stage_C -->
## Stage C
- kit_sha: 648b990 | seed: 42 | fit rows: 100000 | predict rows: {'train': 363595, 'holdout': 122006, 'capped': False}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 26.7 min | total: 30.3 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.910, event holdout 0.911, player train 0.952, player holdout 0.956
- calibration — weighted ECE 0.0422
- ELPD (lppd) -80107.5 ± 243.6 over 122006 events
- undercorrection corr — model 0.031 vs public 0.013
- localization slopes (per ft/s) — grounder 0.0023, barrel 0.0009
- sanity warnings: ['max R-hat on probed mu cells = 1.174 (> 1.1)']
<!-- /stage_C -->

## v0 outcome

- Player-season replication (100+ PA): Pearson r 0.960 (Stage B) / 0.956 (Stage C) vs
  public Savant xwOBA, with full posterior credible intervals per player.
- Event-level replication vs `estimated_woba_using_speedangle`: r ~0.91 (the model adds
  a sprint dimension the public estimate omits and is a discretized BART surface, so
  <1.0 is expected).
- Calibration on the 2025 holdout: class-frequency-weighted ECE ~0.038–0.042; every
  class including triples (ECE ~0.002) is calibrated with no probability collapse.
- Sprint-speed localization is directionally correct: on weak/topped contact
  (launch_speed_angle 1–2) predicted value rises with sprint speed (slope +0.0024/ft·s⁻¹);
  on barrels (5–6) it is flat (~0). Variable importance: launch_angle > launch_speed >
  sprint_speed.
- ELPD anchor (holdout, lppd): −80107 ± 244 over 122,006 events — the fixed target v1+
  must beat.
- Open finding: the §9.4 speed-undercorrection test is inconclusive. Public's
  residual-vs-sprint correlation on holdout ground balls is only ~+0.013, so there is
  little undercorrection to shrink; the localization check is the cleaner evidence that
  the model captures the speed effect.

## Post-v0 benchmark — is v0 more accurate than Savant? (parity)

Canonical xwOBA-validity test (no re-fit; `scripts/benchmark_vs_savant.py`,
`results/benchmark/`): does year-T xwOBA predict year-(T+1) **actual** wOBA better?
1,058 player-pairs (100+ PA both years), pooled Pearson r vs next-season wOBA —
v0 model **0.481**, Savant **0.487**, naive last-year wOBA 0.390. Paired bootstrap
`r_model − r_savant = −0.006`, 95% CI [−0.025, +0.013] → **statistical parity;
v0 does not beat Savant** (both clearly beat the naive baseline, so v0 is genuine
xwOBA). The 3-feature model is at Savant's information ceiling — beating Savant
requires inputs Savant lacks (→ v1 spray + handedness). **v1 target: pooled r > 0.487
vs next-season actual wOBA with a gap-CI excluding 0.** See `results/benchmark/NOTES.md`.
The per-player credible intervals are honest but do NOT shrink with PA (Task A,
`results/task_a/`) — they track surface uncertainty, not sample size.

## Post-v0 — sample-size-aware per-player interval (the actual product)

If the deliverable is a per-player band showing where a hitter's xwOBA could sit given
their N PAs (wide early, tightening with PA), v0's model interval is the wrong object.
A **bootstrap over each player's PAs** (`scripts/player_ci_bootstrap.py`,
`results/player_ci/`, no re-fit) gives the right band: median width falls 0.102 (100–150
PA) → 0.051 (500–750 PA), `log(width)`~`log(PA)` slope −0.42, width vs analytic SE r=0.994.
It **crosses v0's flat model band at ~400 PA**: below that v0 is up to ~1.8× too narrow
(0.056 vs 0.102 at 100 PA) — it understates uncertainty exactly in the short-term / small-
sample regime the product targets. This band is model-agnostic (built on Savant per-BBE
values). The principled end-state combines two terms, `width ≈ √(sampling² + surface²)` —
the bootstrap sampling term (dominant at low PA) plus BART's surface posterior (~flat 0.056,
dominant at high PA), which is the one piece BART uniquely supplies. See
`results/player_ci/NOTES.md`.

## Talent estimates (empirical Bayes) — true-talent xwOBA + calibrated interval

The object for *analyzing a hitter* (not just banding the raw number): a per-batter-season
xwOBA regressed for sample size, with an interval that narrows with PA. Gaussian–Gaussian
empirical Bayes over quantities we already have (`src/talent.py`, `scripts/run_talent.py`,
`results/talent/`, no re-fit): `θ̂ = μ + reliability·(raw − μ)`,
`reliability = τ²/(τ²+se²)`, 90% interval `θ̂ ± 1.645·√(reliability·se²)`. Hyperparameters
fit **per season** on PA ≥ 100 (μ ≈ 0.305–0.318, τ ≈ 0.031, median reliability ≈ 0.65),
applied to all 2,636 player-seasons. Small samples pull toward the mean (Trout 2024, 125 PA:
raw .407 → **talent .341** [.299,.383]); interval width falls 0.083 (30–100 PA) → 0.047 (450+).

**Validation (predict next-season actual wOBA, r; target-year PA ≥ 100):**

| population | n | EB talent | raw | Savant |
|---|---|---|---|---|
| pooled, PA_T ≥ 100 (vs 0.487 anchor) | 1060 | **0.489** | 0.484 | 0.491 |
| pooled, PA_T ≥ 30 (admits low-PA) | 1173 | **0.467** | 0.445 | 0.452 |

**Verdict: EB talent beats raw in both populations and beats Savant once genuinely low-PA
seasons are admitted (0.467 vs 0.452); at the PA≥100 anchor it is at parity with Savant
(0.489 vs 0.491)** — same tie as v0, but now with a sample-size-honest *center*. The win is
a **pooled** variance-compression effect, not a per-band one: within a narrow PA band
reliability is ~constant so `θ̂` is ~affine in raw and Pearson r is affine-invariant — the
per-band talent−raw gaps (`by_band` in `talent_metrics.json`) are noise, **not** a bug.
Carried to Phase 2: the prior is the flat league mean (→ BART contact-quality prior), and
the interval is estimation-only (→ combine with BART's surface term). See
`results/talent/NOTES.md`.

## Level-2 talent model (joint MVN over xwOBA + peripherals)

Phase 2 / Stage 1. Phase 1 shrinks every hitter toward the **season league mean**; Level 2
shrinks him toward **what his contact quality implies**. The three stats are modeled as jointly
noisy measurements of correlated latent talents — `z_i = (xwOBA, avg EV, barrel rate)_i ~
N((θ_i,ξ_i), S_i)`, `(θ_i,ξ_i) ~ N(μ_season, Σ_talent)` — so the posterior
`θ̂ = μ + Σ(Σ+S_i)⁻¹(z−μ)` leans on the fast-stabilizing peripherals exactly when the xwOBA
sample is small. `S_i` is bootstrapped from the player's own PAs and **keeps its off-diagonals**
(all three stats come from the same balls; ignoring that shared noise is what would manufacture
fake low-PA gains). Closed-form throughout — no MCMC, no BART re-fit, ~7 s
(`src/talent2.py`, `scripts/run_talent2.py --stage full`, `results/talent2/`). Fitted talent
correlations: xwOBA/EV **+0.776**, xwOBA/barrel **+0.712**; xwOBA talent SD 0.0312 (Phase-1 τ
0.0307–0.0323). 2,543 of 2,636 player-seasons use all three dims; 93 fall back to 1-D.

**Validation (predict next-season actual wOBA, r; target-year PA ≥ 100):**

| population | n | **Level 2** | Phase-1 talent | raw | Savant |
|---|---|---|---|---|---|
| pooled, PA_T ≥ 100 | 1060 | **0.4908** | 0.4886 | 0.4835 | 0.4908 |
| pooled, PA_T ≥ 30 | 1173 | **0.4698** | 0.4669 | 0.4454 | 0.4521 |

**Verdict: G3 (low-PA win) and G4 (high-PA non-inferiority) both PASS** — Level 2 beats Phase 1
on r and calibrated RMSE at PA ≥ 30, beats Savant by 0.0177 there, and at PA ≥ 100 now *ties*
Savant (0.4908) where Phase 1 trailed. The gain sits exactly where the design predicted, by PA
band: **+0.0718** (30–60 PA), +0.0365 (60–100), +0.0166 (100–250), −0.0000 (250+). Unlike Phase
1's per-band table, these gaps are real signal rather than affine-invariance noise — Level 2
adds independent information inside the band.

**But it is not statistically established, and the notes say so.** The paired bootstrap
(5,000 reps, PA ≥ 30) gives Δr **+0.0029, CI95 [−0.0117, +0.0176]**, better in 64% of resamples;
and the held-out confirmation season **reverses sign** (select 22→23 + 23→24: +0.0063; confirm
24→25: −0.0034). Nothing was tuned on confirm to rescue it. Ships as an improvement in
expectation, not a demonstrated one.

**Gate outcomes.** G1 (reproduces Phase 1) PASS — corr **0.99950** on the 2,500 rows where both
models should agree, anchors reproduced to 0.4885/0.4886 and 0.4663/0.4669. G2 (bootstrap vs
analytic SE) PASS — corr 0.9965, median ratio 1.002. G3, G4 PASS. **G5 (shared-noise tripwire)
CLEAN and decisive**: refitting with `S_i`'s off-diagonals zeroed does not inflate the gain, it
destroys it (+0.0029 → **−0.0101**, worse than Phase 1; artifact gap −0.0130 against a +0.005
alarm threshold) — the correlated-noise modeling is load-bearing and the win is not an artifact.
G6 (honest split) followed; the disagreement is reported above, not dropped.

Also closes **Phase-1 limitation 3**: a hitter with 2 PA and 2 outs has zero sample variance, so
Phase 1 reported his true talent as exactly .000 with full confidence (Trevor Rogers, a pitcher,
2025). A `(0.25)²/n` variance floor fixes 136 such rows; they are excluded from G1's correlation
and reported separately under `l2a.floor_fix` rather than folded into a pass. See
`results/talent2/NOTES.md`.

## Stage C decision (spec §15.3)

From Stage B (14.2 min on 50k rows), full-train (~363k rows) extrapolates to ~100 min
fit and ~14.5 GB of in-memory `mu` on an 18 GB machine (OOM risk, spec §7.2). The
config-default **100k stratified subsample** was chosen (~26.7 min, ~4 GB). Stage C's
metrics match Stage B's within noise (player r 0.956 vs 0.963, event r 0.91, ECE 0.042,
ELPD −80107 vs −80024), so the three-feature model is quality-saturated by ~50k rows;
the extra data only tightened R-hat (1.67 → 1.17).

## Deviations from spec/plan (spec §11)

- Resolved dependency versions are newer than the plan assumed: pymc 6.1.0,
  pymc-bart 0.12.0, arviz 1.2.0, polars 1.42.1, pandas 3.0.3, numpy 2.4.6,
  pytensor 3.1.3. Frozen in `requirements.lock`.
- `src/config.py`: `.env` is resolved with `find_dotenv(usecwd=True)` so production picks
  up the repo `.env` (runs from repo root) while tests that chdir to a tmp dir stay
  hermetic.
- `src/prep.py` `stratified_subsample`: largest-remainder ties break on class index
  (deterministic) — polars `group_by` is not order-stable, which made the subsample
  non-reproducible.
- `scripts/smoke_model.py`: executable code is under `if __name__ == "__main__":` —
  pymc-bart's `Manager` and pymc's parallel chains use the spawn start method (macOS
  default) and re-import the module. Synthetic HR signal strengthened to `0.20·(EV−95)`
  so the tiny-fit assertion is robust to spawn's non-reproducible RNG.
- OOS prediction (spec §7.3): the documented `pm.set_data` + `sample_posterior_predictive`
  path silently FREEZES `mu` in pymc-bart 0.12 (ImplicitFreezeWarning; returns the
  in-sample trace of shape (K, n_train) regardless of X_new — a wrong answer, not an
  exception). The spec-sanctioned stored-trees predictor (`_sample_posterior`) is the
  live path; its output is (S, n, K), transposed to the (S, K, n) convention.
- `verify_oos_mechanism` gate: changed from `max_abs_diff < 0.05` to `corr > 0.99 AND
  mean_abs_diff < 0.03`. The stored-trees predictor averages a random subset of trees
  vs a different thinned in-sample subset; for a high-variance BART posterior the max
  abs diff is Monte-Carlo noise that grows with event count (0.03 at 800 events, 0.19 at
  2000), not a mechanism error. Corr (0.996–0.998) and mean abs diff (0.008–0.012) are
  scale-invariant and catch real errors.
- NetCDF I/O: arviz 1.2 ships no NetCDF backend; added `h5netcdf` + `h5py` for
  `idata.to_netcdf`.
- `variable_importance`: `compute_variable_importance` returns large per-event
  `preds`/`preds_all` arrays; only the small summary fields are serialized so
  `metrics.json` stays ~14 KB (was ~194 MB).
- BART `mu` R-hat is a warning at every stage (was a hard stop for B/C): mu-cell R-hat is
  structurally high (the sum-of-trees is not identified at the cell level; 1.83 → 1.67 →
  1.17 as data grows) and is not a meaningful convergence signal — `verify_oos_mechanism`
  (corr ~1.0) gates real convergence. Approved at the Stage A review gate.
