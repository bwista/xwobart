# xwobart v0 results

<!-- stage_A -->
## Stage A
- kit_sha: fcbb78a | seed: 42 | fit rows: 5000 | predict rows: {'train': 20000, 'holdout': 20000, 'capped': True}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 0.4 min | total: 0.6 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.895, event holdout 0.893, player train 0.471, player holdout 0.506
- calibration — weighted ECE 0.0581
- ELPD (lppd) -13818.7 ± 99.2 over 20000 events
- undercorrection corr — model 0.034 vs public -0.002
- localization slopes (per ft/s) — grounder -0.0002, barrel -0.0042
- sanity warnings: ['max R-hat on probed mu cells = 1.806 (> 1.1)']
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

### Phase 2 Stages 2–3 (2026-07-20)

- `FEATURES` split into `FEATURES_V0` / `FEATURES_SPRAY`; `build_features` takes the
  column list, so **one** orchestrator serves both variants and protocol parity with the
  frozen anchor is structural rather than aspirational (`run_v0.py --variant`).
- `stand` enters BART as a numeric `stand_R` (1.0/0.0) column, not the raw string.
- The design's hc missing-flag is an **audit column, not a BART feature** (0.034–0.043%
  missingness; a flag feature would be split noise and would make it a 6-feature model).
- Spray imputation uses the conditional **median**, not the spec's conditional **mean**,
  because ~9% of BBE sit outside the foul lines in hc coordinates (|φ_raw| > 45°, caught
  fouls) and the mean is sensitive to that tail.
- Level 2 was re-run in **Stage 2, not Stage 3**: `run_talent2.py` consumes public Savant
  per-event xwOBA and never touches the surface, so a refit provably cannot move it — but
  the cache rebuild could, so the re-run belongs to the rebuild's gate.
- **`all_trees` must be materialized before pickling.** pymc-bart stores the fitted trees
  in a `multiprocessing.Manager` ListProxy (`pymc_bart/bart.py:143`), so
  `pickle.dump(all_trees)` writes a ~208-byte connection token that raises
  `FileNotFoundError` once the manager process dies. `pickle.dump(list(all_trees))` yields
  a real, reloadable pickle. Caught by the Stage-A smoke, not by reasoning.
- **Task 8 was split into two processes** (`run_v0.py --variant spray`, then
  `scripts/marginalize_spray.py`), which the pickle fix above makes possible: the plan
  required one 60-minute invocation only because the trees were thought unrecoverable.
  The from-pickle prediction path was verified against the in-process path at corr
  0.99989 / mean|diff| 0.0027 — the same thresholds `verify_oos_mechanism` uses.
- The rollup A/B races the two rollups **directly** against next-season wOBA rather than
  feeding both through Level 2 (spec §"Rollup choice under spray"). Level 2 currently
  consumes public Savant per-event xwOBA, so wiring the model rollup into it is itself
  Stage-4 work; the direct race answers the same question. Stage 4 should confirm the
  choice survives the talent layer.
- `results/rollup_ab/figures/` uses two **lightness steps of the model hue** for the two
  rollups rather than two competing hues. Four separable categorical hues do not exist
  alongside the repo's fixed blue=model / orange=Savant mapping — the obvious 4th (green)
  sits at ΔE 4.0 from Savant's orange under deuteranopia. The two rollups are one entity
  in two configurations, so a lightness pair is also the semantically correct encoding.
  Every bar carries a direct value label (the two low-chroma slots fall below 3:1 contrast
  on white, which obligates visible labels).

## Sampler reproducibility — measured, and it bounds every surface comparison

**pymc-bart 0.12 fits are not reproducible across processes, despite `random_seed=42`.**
Three back-to-back Stage-A runs on bit-identical inputs returned ELPD −14,299 / −14,375 /
−13,819 (a 556-nat spread over 20,000 events). The tell is that **R-hat itself moves**
(1.37 ↔ 1.83): R-hat is a deterministic function of the draws given a fixed probe seed, so
the *draws* differ, not the inputs. Input identity was proved separately — `X_fit` is
bit-equal at both the 5,000- and 100,000-row subsample sizes, and the rebuilt caches match
the pre-rebuild backup row-for-row.

At the scale that matters, the noise is much smaller. A **v0 Stage-C replicate** — identical
code, identical inputs, run in the same session as this table:

| | holdout ELPD (122,006 events) |
|---|---|
| frozen v0 anchor | **−80,107.5** (SE 243.6) |
| v0 replicate | **−79,840.4** (SE 245.8) |
| run-to-run delta | **+267.1 nats** = 1.10 × anchor SE = +0.0022 / event |

Consequences, and they are not optional reading:

1. **Gate E1's ≥1000-nat bar is 3.74× the measured null noise**, so it survives — a gain
   that size cannot be a lucky draw.
2. **The frozen anchor sits ~267 nats on the unlucky side of a fresh run.** Any surface
   delta should therefore be reported against *both* the anchor and a same-session control,
   not the anchor alone.
3. **The grounder sprint slope is a noisy statistic** — 0.0023488 → 0.0009895 between two
   identical runs. Do not over-read the E7 localization diagnostic.
4. Never conclude "a refactor broke the v0 path" from moved metrics alone. Prove input
   identity instead: `X_fit` bit-equality plus `fit_rows` / `predict_rows` / class
   distribution / linear weights. The replicate also confirmed the `--variant` refactor at
   full scale (fit rows 100,000; predict rows 363,595 / 122,006; VI indices `[1, 0, 2]` —
   all identical to the anchor).

## Phase 2 Stage 2 — spray cache rebuild + sign QC (COMPLETE)

Full write-up: `results/stage2_rebuild/NOTES.md`. Reproduce with
`scripts/rebuild_caches.py` then `scripts/qc_spray.py`.

The slim caches now carry `hc_x`, `hc_y`, `stand` (15 → 18 columns), and
`src/prep.add_spray` derives `spray_pull = ±atan2(hc_x − 125.42, 198.27 − hc_y)`, mirrored
by `stand` so **positive always means pulled** for both hands.

**Reproduction gates R1–R6: all PASS.** The one that matters most is R2 — the
order-independent content digest over the pre-existing 15 columns came back
**byte-identical on all four seasons**, so the upstream KIT cache had not moved and every
frozen anchor in this repo survives the rebuild. R4 re-derived the Phase-1 talent table at
`max|Δ|` of exactly **0.0** over 2,636 rows; R5 held BBE counts at 118,891 / 122,070 /
122,634 / 122,006 (⇒ v0's 363,595 train + 122,006 holdout, so the ELPD anchor stays
comparable); R6 reproduced Level 2 at r 0.469817 / 0.490783 to six decimals.

**Sign-QC gates S1–S6: all PASS**, reproducing every planned anchor exactly — league mean
pull L +6.84…+7.50 / R +3.23…+3.62, HR mean +16.3…+20.3 both hands, ~78–84% pull-side,
Schwarber +13.90, Paredes +11.88, Mountcastle −5.16. S5, the gate that catches a
modal-hand (rather than per-event) mirror, reports **+8.68 L / +5.60 R** on the 65
switch-hitter batter-seasons — the correct-mirror row, and far from the simulated bug's
−3.97.

## Phase 2 Stage 3 — spray surface: **E1 FAILS. Spray does not improve the surface.**

`results/stage_C_spray/` · 5 features `[launch_speed, launch_angle, spray_pull, stand_R,
sprint_speed]` · same orchestrator, same 100,000-row seed-42 subsample, same 122,006-event
holdout as v0 Stage C · fit 26.2 min, total 30.0 min.

### The verdict

| | holdout ELPD (122,006 events) |
|---|---|
| frozen v0 anchor (3-feature) | −80,107.5 |
| v0 replicate (3-feature, same session) | −79,840.4 |
| **spray (5-feature)** | **−79,876.3** |

- **spray − anchor = +231.2 nats.** E1 required **≥ +1000**. → **FAIL.**
- **spray − v0 replicate = −35.9 nats.** Against a same-session 3-feature control the
  spray surface is, if anything, *slightly worse*.
- The apparent +231 "gain" is **inside the measured +267.1-nat run-to-run noise floor** —
  it is fully explained by the frozen anchor being an unlucky draw, not by spray.

Without the replicate this would have read as a real-but-sub-threshold improvement. It is
not an improvement at all. **That is the whole reason the noise floor was measured first.**

### The implementation is correct — this is a genuine negative, not a bug

The plan's 5-item E1-failure checklist comes back clean on every item:

1. `spray_pull` is in `X_fit` — and pymc-bart's own variable importance ranks it **#3 of 5**
   (`indices [1, 0, 2, 3, 4]`: launch_angle > launch_speed > **spray_pull** > stand_R >
   sprint_speed).
2. Sign QC ran against the rebuilt caches: mean `spray_pull` on the fit frame is **+7.13
   (LHB) / +3.47 (RHB)** — both positive, matching the S1 anchors.
3. `stand_R` is not constant: values {0.0, 1.0}, mean **0.5935** (the league RHB share).
4. Holdout event count is exactly **122,006**.
5. Subsample is **100,000 rows at seed 42**; predict rows 363,595 / 122,006; hc imputation
   0.039% train / 0.036% holdout.

And the model plainly *learned* spray: `figures/pdp_la_spray_hr.png` shows the HR band at
LA 25–35° leaning strongly to the pulled side, P(HR) rising **0.20 → 0.37** from −40° (oppo)
to +40° (pull) at fixed EV 103. The feature is real, correctly signed, and used. It simply
does not buy out-of-sample likelihood.

### Other gates

| gate | result | detail |
|---|---|---|
| **E1** ELPD ≥ anchor + 1000 | **FAIL** | +231.2 nats, inside the 267-nat null noise |
| **E2** ECE ≤ 0.046456 | **FAIL** | **0.053134** vs v0 0.042233 / replicate 0.040203 |
| E3 triple Brier ≤ 0.005013 | PASS (diagnostic) | 0.005007; double improves (0.050683 vs 0.052073), HR worsens (0.027107 vs 0.025344) |
| E4 `verify_oos_mechanism` | PASS | corr 0.9969, mean abs diff 0.0106 |
| E5 no collapsed classes | PASS | only the structural mu R-hat warning (1.188) |
| E6 draws persisted | PASS | (200, 363,595) + (200, 122,006) float32, row-aligned |
| E7 importance / PDP | mixed (diagnostic) | spray_pull top-3 ✓, HR-band PDP leans pulled ✓, but sprint migration is **backwards**: pulled-grounder slope 0.000716 < oppo 0.000883 (`pull_minus_oppo` −0.000167) |

E2 is the second hard failure and it sharpens the story: the 5-feature surface did not
trade calibration for likelihood — **it lost on both**. Event replication also falls
(r 0.8734 vs 0.9105 anchor / 0.9185 replicate) and player replication with it (0.9075 vs
0.9560). The plan anticipated a small replication drop *alongside an ELPD gain*; with no
gain, this is just degradation.

### What this means, and what was deliberately not done

Per the plan, no hyperparameter was touched after the failure — raising `m_trees`, `draws`
or the subsample would break the protocol parity that makes the anchor comparison valid at
all. A documented non-beat is a valid outcome of this plan; a rescued one is not.

The most likely reading is **capacity dilution, not missing information**: with `m_trees`
fixed at 50 and the same 100,000 training rows, two extra dimensions spread the same tree
budget over a larger feature space, and the splits spent resolving spray are splits no
longer spent resolving EV × LA. The in-sample PDP shows the signal is there; the held-out
likelihood shows it does not pay for what it costs. That yields a clean, testable follow-up
for a future stage — **match capacity (raise `m_trees`) and re-run both variants under the
new setting**, comparing spray-vs-v0 at equal capacity rather than against a frozen
3-feature anchor. That is a new experiment with its own anchor, not a rescue of this one.

### Rollup A/B (E8) — design risk 2 is CONFIRMED

`results/rollup_ab/` · full write-up in `results/rollup_ab/NOTES.md`. Season-T rollup vs
season-(T+1) **actual** wOBA, calibrated RMSE (×10⁻³, lower is better):

| pool | n | conditioned | marginalized | v0 (3-feature) | Savant |
|---|---|---|---|---|---|
| PA ≥ 30 | 1,183 | 36.59 | 36.27 | **35.82** | **35.56** |
| PA ≥ 100 | 1,072 | 35.44 | 35.07 | **34.59** | **34.45** |

**Conditioning the rollup on per-ball spray direction is reliably counterproductive.**
Marginalizing spray out (9 equal-mass league quantiles per EV × LA × stand cell) beats
conditioning in 7 of 8 band/pair splits; the paired bootstrap (5,000 reps, seed 42) puts
`conditioned − marginalized` at **+0.000317** [+0.000189, +0.000458] at PA ≥ 30 with
conditioned winning **0 of 5,000** resamples. The *direction* is as resolved as a bootstrap
can make it; the *magnitude* (~0.0003) is below this plan's 0.001 practical bar. Both
statements hold — report them together.

So: prefer **marginalized**. But the choice is academic here, because **both spray rollups
lose to v0 in every band and every season pair**, which is exactly what E1's failure
predicts. There is no spray rollup worth promoting into the talent layer, and Stage 4's
premise as written (push the A/B winner through Level 2) needs rethinking first.

The design also predicted a descriptive/predictive **inversion** (conditioned describes
better, predicts worse). Not observed: same-season correlation with Savant at PA ≥ 100 is
conditioned 0.901 < marginalized 0.921 < v0 0.948 — conditioned agrees least *and* predicts
worst. Caveat on reading that: Savant is spray-blind, so a spray-conditioned rollup drifts
from it mechanically; low agreement is not itself evidence of worse description. Testing the
inversion properly needs same-season *actual* wOBA as the target — a different experiment.

### Persisted per-event draws (E6) — location, contract, and a caveat that must travel with them

`results/stage_C_spray/` (all gitignored — ~390 MB of draws plus a 27.6 MB trees pickle):

| file | shape | meaning |
|---|---|---|
| `ev_draws_{train,holdout}.npy` | (200, 363,595) / (200, 122,006) f32 | spray-conditioned per-event value draws |
| `ev_marginalized_{train,holdout}.npy` | (363,595) / (122,006) f32 | spray-marginalized per-event values |
| `ev_draws_keys_{train,holdout}.parquet` | 363,595 / 122,006 rows | `row`, `batter`, `season`, `woba_denom` + the 5 features + `hc_imputed` |
| `lppd_i_holdout.npy` | (122,006) f64 | per-event holdout log-likelihood |
| `all_trees.pkl` | 27.6 MB | materialized fitted trees (see the deviation note) |

**Alignment is positional**: axis 1 of `ev_draws_{tag}.npy` ↔ the row order of
`ev_draws_keys_{tag}.parquet`, whose `row` column is that index. `metrics.json` also stamps a
`batter_order_digest` per split so a reordering is detectable rather than merely
contractual. When concatenating train + holdout, drop `row` first — it restarts at 0 in each
file.

**Surface-uncertainty caveat (spec Risk 3) — this must travel with the `.npy` files, because
Stage 4 is a separate plan and a separate session.** The between-draw variance of these
arrays is exactly right for *per-player* intervals. It is **not** valid for league-aggregate
claims: surface errors are **correlated across players in the same feature region**, so
summing or averaging these intervals across players understates the true uncertainty. A
league-level statement needs the per-draw refit variant, not these draws.

### Next experiment — does spray help at matched capacity? (`scripts/capacity_experiment.py`)

The capacity-dilution reading above is testable, and the script is written and smoke-tested
end to end (not yet run at Stage C). It fits **three** models at the same enlarged tree
budget and compares them to each other rather than to the frozen 3-feature anchor:

```bash
.venv/bin/python scripts/capacity_experiment.py --dry-run        # print the plan
.venv/bin/python scripts/capacity_experiment.py                  # m_trees=200, ~5.3 h
.venv/bin/python scripts/capacity_experiment.py --analyze        # re-score existing fits
.venv/bin/python scripts/capacity_experiment.py --stage A --m-trees 8   # ~1 min wiring smoke
```

1. **v0 at the new capacity** — the fresh anchor.
2. **v0 again** — the replicate, whose gap to (1) *measures* the noise floor at this
   capacity rather than assuming Stage 3's 267 nats carries over.
3. **spray at the new capacity** — the treatment.

Completed fits are skipped on re-run, so an interruption costs one fit rather than the set
(each is ~107 min at `m_trees=200`, and each writes ~4 GB of `idata.nc`).

Two verdicts are reported. The **unpaired** one repeats Stage 3's logic against the new
anchor and noise floor. The **paired** one is sharper and is why `run_v0.py` now persists
per-event holdout log-likelihood for *every* variant: over the same 122,006 events in the
same order it bootstraps the per-event difference, and because both models are driven
mostly by the same EV × LA signal their per-event errors are strongly correlated — so the
paired interval is far tighter than the ±244 on either total and can resolve differences
the unpaired test cannot see. `metrics["holdout_order_digest"]` is asserted equal across
runs before pairing, so the test cannot silently compare misaligned events.

Both verdicts distinguish three outcomes — better, worse, indistinguishable. A large
*negative* gap is a significant degradation, not a null result.

**Localization is not like-for-like under `--variant spray`.**
`metrics["localization"]["grounder_slope_per_ftps"]` is the **pulled**-grounder slope (the
spray grids are RHB at ±20°), so it cannot be compared directly to v0's spray-blind
0.0023488. And per the reproducibility section above, that slope moved 0.0023488 → 0.0009895
between two *identical* v0 runs — it is too noisy to carry an argument either way.

<!-- stage_A_spray -->
## Stage A_spray
- kit_sha: fcbb78a | seed: 42 | fit rows: 5000 | predict rows: {'train': 20000, 'holdout': 20000, 'capped': True}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 0.4 min | total: 0.8 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.819, event holdout 0.813, player train 0.459, player holdout 0.465
- calibration — weighted ECE 0.0559
- ELPD (lppd) -14561.7 ± 105.1 over 20000 events
- undercorrection corr — model 0.027 vs public -0.002
- localization slopes (per ft/s) — grounder 0.0019, barrel 0.0003
- sanity warnings: ['max R-hat on probed mu cells = 1.316 (> 1.1)']
<!-- /stage_A_spray -->

<!-- stage_C_spray -->
## Stage C_spray
- kit_sha: fcbb78a | seed: 42 | fit rows: 100000 | predict rows: {'train': 363595, 'holdout': 122006, 'capped': False}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 26.2 min | total: 30.0 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.874, event holdout 0.873, player train 0.909, player holdout 0.908
- calibration — weighted ECE 0.0531
- ELPD (lppd) -79876.3 ± 238.2 over 122006 events
- undercorrection corr — model 0.042 vs public 0.013
- localization slopes (per ft/s) — grounder 0.0007, barrel -0.0023
- sanity warnings: ['max R-hat on probed mu cells = 1.188 (> 1.1)']
<!-- /stage_C_spray -->
