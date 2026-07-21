# Rest-of-season xwOBA forecast — design spec

> Written 2026-07-20. Design of record for the next xwobart deliverable. Supersedes the
> Phase-2 **Stage 4** handoff (`docs/superpowers/handoffs/2026-07-20-stage4-brainstorm-handoff.md`),
> which scoped a *true-talent coverage interval* built on the persisted surface draws. This
> spec reframes the product after a design conversation; see **§1**. No code here — spec only.

## 1. Why this, and not Stage 4 as handed off

The Stage 4 handoff aimed to fold the BART **surface** uncertainty (the Stage-3 per-event value
draws) into the Level-2 talent interval and validate *coverage of true talent*. Two problems
surfaced when we worked the decisions:

1. **The truth is unobservable and every proxy is confounded.** True talent can only be checked
   against next-season wOBA (conflated with aging drift + next-season sampling noise) or a
   within-season PA split. And a within-season split with a *shared* surface is **blind to the
   surface term** — a surface mis-valuation shifts both halves of the split identically and
   cancels, so the split cannot validate (and would actively penalize) the very term Stage 4
   adds. The surface term matters most exactly where it is hardest to validate (high PA, where
   sampling has shrunk away).
2. **It was not the product the user wanted.** The actual deliverable is: *a current best
   estimate of a hitter's xwOBA talent (the **descriptor**) plus a range for where his xwOBA
   will **finish the season** (the **short-term forecast**), both sharpening as the season
   accrues PAs, and allowed to use his multi-year history.*

Reframing to "forecast the final full-season xwOBA from a mid-season cutpoint" dissolves both
problems at once:

- **The truth becomes observable and unconfounded.** "Rest of season" is the *same* season — no
  aging, no next-year environment — and it actually happens, so we score the forecast against the
  realized final line. Every (player, mid-season cutpoint) pair is a test case.
- **The surface term cancels by construction.** The target is an *xwOBA number* (computed on a
  fixed value surface), so our uncertainty about how to *value* a ball sits identically in the
  forecast and the target and drops out. Surface uncertainty is **out of scope**; the Stage-3
  draws are **shelved** (see §10).

What remains is the lever the design review ranked #1 all along and never built
(`2026-07-19-xwobart-phase2-design-response.md` §5.1; `results/talent2/NOTES.md` limitation 1):
**a hierarchical player model that pools a hitter's own seasons** to sharpen his current-season
talent estimate — most valuable early in the year when in-season data is thin.

## 2. Product definition

For a hitter *i* in season *t*, standing at a cutpoint after his first *k* plate appearances:

- **Descriptor** `θ̂_{i,t,k}` — the posterior-mean estimate of his true per-PA xwOBA-value talent
  this season, informed by (i) his first *k* in-season PAs, (ii) his own completed prior seasons
  (career pooling + aging), (iii) his peripherals (a rung).
- **Forecast** — a predictive interval for his **final full-season xwOBA** `r_final`, reported at
  50 / 80 / 90%, narrowing as the season runs out.

Both come from one posterior. "Final full-season" = option **A** from the design chat (where the
season line *finishes*, with PAs-so-far locked in), not the standalone rest-of-season rate.

## 3. The core decomposition (what we actually have to model)

Each PA *j* carries a wOBA value `v_j` (batted ball → `estimated_woba_using_speedangle`, else
`woba_value`) and denominator `d_j = woba_denom`. A rate over a PA set *P* is
`r(P) = Σ_P v_j / Σ_P d_j`, exactly as `src/talent.py` / `talent2.py` build it.

At cutpoint *k*, split the season into observed *O* (`|O| = k`, denom `D_obs`) and remaining *R*
(`m` PAs, denom `D_rest`). The final line is a **known-weight blend**:

```
r_final = (1 − w)·r_obs  +  w·r_rest ,      w = D_rest / (D_obs + D_rest)
```

`r_obs` is **known** (locked in). Only `r_rest` (the rest-of-season rate) is uncertain, so the
forecast error is

```
error(r_final) = w · (r̂_rest − r_rest^actual).
```

**Consequences that shape the whole design:**

1. The **only quantity to model is the rest-of-season rate `r_rest`.** The final-line product is
   that forecast down-weighted by *w*. The descriptor is the center of `r_rest`.
2. The interval **narrows automatically as the season ends** (`w → 0`), on top of narrowing
   because talent is better estimated. Both requested axes fall out for free.
3. Validation of the final line is equivalent (up to the known scale *w*) to validation of the
   rest-of-season-rate forecast — a clean, same-season, unconfounded target.

## 4. The model

### 4.1 Latent talent, across a player's seasons

```
θ_{i,t}  =  μ_t  +  g(age_{i,t}; β)  +  η_i  +  u_{i,t}
```

- **μ_t** — the **per-season league environment** (per-PA xwOBA-value), **not** a global scalar.
  League offense drifts materially year to year (Phase 1's per-season μ runs 0.305 → 0.318 across
  2022–25, a ~0.013 swing — larger than the effects the model chases), so a global mean would
  mis-center every forecast and specifically handicap it against the naive benchmark (which is
  season-*t*-correct by construction — see G1). μ_t is estimated **causally from within-season
  league xwOBA available at the cutpoint** (leaking nothing about the rest of season *t*), so it is
  **not** a LOSO-fit hyperparameter (§4.3, §7); exact estimator deferred to §14.
- **g(age; β)** — a shared aging curve (quadratic centered near the peak age, `β₂ < 0`; spline as
  a refinement). **Aging is a rung** — it needs external birthdates (§10), and is *additive*, so
  the base model runs without it.
- **η_i ~ N(0, σ_η²)** — the player's career-level random intercept. **This is what carries
  multi-year history**: it is estimated from his completed prior seasons and pulls his
  current-season prior toward his own norm rather than the league mean.
- **u_{i,t} ~ N(0, σ_u²)** iid — season-to-season deviation left after career level and aging.
  **AR(1)** (`u_{i,t} = ρ·u_{i,t−1} + e`) is a refinement rung so last season informs this one
  beyond the career mean.

When a player has no prior seasons and aging is off, this reduces to a single measurement shrunk
toward μ_t — i.e. **exactly Phase 1 / Level 2** (the basis of gate G5).

### 4.2 Measurement (within-season, truncated)

Each observed sample is a **noisy read** on that season's talent:

```
z_{i,s}  =  θ_{i,s}  +  ε_{i,s},      ε_{i,s} ~ N(0, S_{i,s})
```

- For **completed prior seasons** *s*, `z_{i,s}` and its sampling variance `S_{i,s}` come from the
  full season.
- For the **target season at the cutpoint**, `z_{i,t,k}` and `S_{i,t,k}` come from the **first
  *k* PAs only** (a wider measurement).

`S_{i,s}` is the bootstrap sampling variance of the rate — `src/talent2.py:bootstrap_S`'s
`S[0,0]`, floored at `FLOOR_SD_PER_PA²/n`. This is **reused verbatim**, run on the relevant PA
subset (the xwOBA-only base rung consumes only `S[0,0]`; `bootstrap_S` still takes `ev`/`barrel`
arrays and returns a 3×3, so the base rung passes those as NaN — the `S[0,0]` branch runs
independently, `talent2.py:100-102`). Structurally the model is a **linear mixed model with known,
per-observation measurement error** (career random intercept + fixed aging + residual drift +
heteroskedastic known noise).

**Peripherals (rung b).** Extend the within-season measurement to the Level-2 joint MVN over
(xwOBA, avg EV, barrel) so the peripheral-informed θ measurement feeds the hierarchy. The exact
composition that avoids double-shrinkage (peripheral shrink at the measurement layer *and* career
shrink at the hierarchy layer) is an **open implementation detail deferred to rung b** (§6); the
base rung is xwOBA-only and is what establishes the multi-year lever.

### 4.3 Fitting — Empirical Bayes (primary)

Two kinds of parameter enter, and they are fit differently:

- **Season environment `μ_t`** — estimated *causally* per season from within-season league
  xwOBA available at the cutpoint (§4.1, §14), **not** by MLE and **not** held out. This keeps the
  prior centered on the right offensive environment mid-season *t*, and is why LOSO (§7) applies
  only to the structural parameters below (a per-season mean cannot be formed for a fully-held-out
  season).
- **Structural hyperparameters `φ = (β, σ_η², σ_u²[, ρ])`** — the aging curve and variance
  components. Everything is linear-Gaussian, so the marginal likelihood of a player's measurements
  `{z_{i,s}}` — integrating out `(η_i, {u_{i,s}})` with the `μ_t` offsets held fixed — is Gaussian.
  Fit `φ` by maximizing the summed marginal log-likelihood over players (L-BFGS, in the style of
  `talent2.py:mvn_mle`). These are population-structure parameters that barely drift season to
  season, so they are the ones LOSO protects. Full Bayes over `φ` (MCMC, propagating hyperparameter
  uncertainty into every interval) is a **stretch goal**, run on a subsample as a robustness
  check — not the primary path.

### 4.4 Per-cutpoint posterior — closed form (the load-bearing trick)

Given `φ`, the posterior of `θ_{i,t}` conditioning on the player's **causally available** data —
his completed prior seasons plus his first-*k* current PAs — is a **closed-form Gaussian**
(linear-Gaussian conditioning; a small per-player Kalman/GMRF solve), like `mvn_posterior`
extended across seasons:

```
θ_{i,t} | (prior seasons, first-k current)  ~  N(θ̂_{i,t,k},  V_{i,t,k})
```

`θ̂` is the **descriptor**; `V` is the **estimation variance**. No MCMC per cutpoint — this is what
makes sweeping thousands of (player, cutpoint) pairs feasible, and it is the same modular "cut
inference" the design review endorsed.

**Causality:** condition only on seasons **s < t** plus the first *k* of *t*; never on *s > t* or
the rest of *t*. Players with no prior in-window season fall back to population + peripherals.

## 5. Range construction (the two pieces)

The rest-of-season rate combines **estimation** uncertainty (`V`) and **future-sample** noise over
the *m* remaining PAs. By simulation, for `b = 1..B`:

1. draw `θ^(b) ~ N(θ̂, V)`  — estimation;
2. draw `r_rest^(b)` = a rate over *m* future PAs given `θ^(b)` — future sampling;
3. `r_final^(b) = (1 − w)·r_obs + w·r_rest^(b)`.

Report empirical quantiles of `{r_final^(b)}` at 50/80/90%.

**Future-sampling (step 2) — forward bootstrap (default).** Resample *m* (value, denom) pairs with
replacement from a reference value distribution to form a rate, then **recenter by an additive
shift** so the resampled-rate distribution has mean `θ^(b)` — an additive shift moves the location
while preserving the shape (the boom/bust asymmetry we want; a multiplicative rescale would distort
it). The reference is
the player's own value multiset (career when available, else league-shaped by his profile for thin
samples). This **preserves the natural asymmetry** (a boom/bust hitter has more upside room — the
real "where could it go" signal, per `results/player_ci/NOTES.md`). **Analytic fallback:**
`r_rest^(b) ~ N(θ^(b), σ_i² / m_eff)` for thin samples, `σ_i²` the per-PA value variance.

**Remaining PAs `m` / `D_rest`.** For **validation**, use the *realized* remaining denom (known
ex-post) — the rate is the object under test, not the schedule. For a **live** forecast, project
`D_rest` from games remaining × the player's PA rate; the rate uncertainty dominates the count
uncertainty, so this is a secondary concern documented but not modeled in v1.

## 6. Rungs (validate incrementally)

| rung | adds | purpose |
|---|---|---|
| **(a) base** | η_i career pooling + iid drift, xwOBA-only | establishes the multi-year lever (G1, G2) |
| **(b) peripherals** | Level-2 joint MVN measurement | low-PA sharpening (resolve double-shrinkage) |
| **(c) refinements** | AR(1) drift and/or spline aging (needs birthdates) | second-order gains |

Ship (a); (b) and (c) are additive and scored on the same test. Each rung reports the full gate
panel so we see where value enters.

## 7. Validation protocol (leakage-safe)

- **Cutpoints.** Stand at `k ∈ {50, 100, 150, 200, 300}` PA. A (player, k) pair is eligible only
  if the player has enough remaining denom to form a real "rest of season" (min-remaining
  threshold, e.g. `D_rest ≥ 30`). Order PAs by `game_date` (day granularity; no intra-game order
  in the cache — acceptable for coarse cutpoints; ties broken stably).
- **Report by two axes:** **PA-seen** *k* (how well talent is known) **and** *w* (how much runway
  remains). Both drive width; a single axis hides the structure.
- **Season environment `μ_t` — causal, per season, not held out.** Estimated from within-season
  league xwOBA at the cutpoint (§4.3, §14); available mid-season, leaks nothing. A per-season mean
  cannot be formed for a fully-held-out season — which is exactly why LOSO applies only to `φ`.
- **Structural hyperparameters `φ` — leave-one-season-out (primary):** to forecast season *t*, fit
  `φ = (β, σ_η², σ_u²[, ρ])` on the other three full seasons. **Strict causal check:** refit `φ`
  from seasons `< t` only; expect a small move (Level 2's analogous convention cost ≈ 0.0005;
  `results/talent2/NOTES.md`).
- **Truth.** The player's *actual* rest-of-season rate → actual `r_final`. Never seen by the model.

## 8. Benchmarks (the bar)

Point forecast of `r_final`, RMSE by PA-seen and *w* band, pooled across eligible pairs, **paired
bootstrap** over players:

1. **naive** — "he keeps hitting as he has": `r̂_rest = r_obs`. *The thing to beat, especially early.*
2. **league-shrunk** — Phase 1 EB on the first *k* PAs.
3. **single-season Level 2** — the current model, no history. *Isolates the multi-year lever.*
4. **Marcel** — a Marcel-style blend of {prior seasons, current-to-date}, weighted + regressed to
   the mean + aged. *The projection bar the machinery must clear.*
5. **Savant-to-date** — Savant's season xwOBA through *k* as the estimate.

## 9. Pre-registered gates

- **G1 — beats naive** on `r_final` RMSE at low PA-seen (paired-bootstrap CI excludes 0). *Core
  claim: history + shrinkage beats "hot start continues."*
- **G2 — beats/ties single-season Level 2.** *The multi-year lever pays.*
- **G3 — beats/ties Marcel.** *The machinery earns its keep over the simple projection.*
- **G4 — calibration** of the 50/80/90 intervals within ±5pp, across PA-seen and *w* bands; report
  sharpness (mean width) at fixed coverage.
- **G5 — reduces to Level 2** when history is removed (strip η_i / prior seasons) **and evaluated at
  the full-season cutpoint with `μ_t` taken as Level 2's per-season league mean**: estimates match
  `results/talent2` to tolerance. Because Level 2 shrinks toward a *per-season* μ_t, this test is
  only well-defined with the per-season environment of §4.1/§4.3 — a global μ would fail it
  spuriously, and the tolerance would silently absorb the gap. *Free regression/sanity test.*

Report **pooled RMSE + a by-band table**, never per-band correlation alone (affine-invariance;
`results/talent/NOTES.md`). Direction-of-improvement claims carry their paired-bootstrap CI.

## 10. Reused / shelved / prerequisites

**Reused:** `src/talent2.py` (`bootstrap_S`, the MVN measurement + `mvn_mle`/`mvn_posterior`
machinery); the `results/player_ci` forward-bootstrap idea; `src/rollup.py`/`talent2.py`
player-season aggregation; the K/BB value splice unchanged for comparability.

**Shelved:** the Stage-3 surface draws (`results/stage_C_spray/ev_draws_*`) and the whole
surface-uncertainty term — the xwOBA-number target makes them cancel (§1). The capacity experiment
remains an independent surface side-quest, unrelated here.

**Data prerequisites (verify in the plan):**
1. **Temporal ordering** — present: `game_date` is in the slim cache (18-col schema). Day
   granularity, no `at_bat_number`; adequate for coarse cutpoints.
2. **Birthdates for aging (rung c only)** — **not** in the cache; needs an external join
   (KIT/Chadwick register, same metadata path as player-name/sprint resolution). Base and rung (b)
   do **not** need it.

## 11. Outputs & code layout

- `src/talent3.py` — pure functions: hierarchical assembly, EB hyper-fit, closed-form per-cutpoint
  posterior, forward-bootstrap forecaster. No orchestration.
- `scripts/run_talent3.py` — orchestration, cutpoint sweep, gate scoring, figures.
- `results/talent3/` — `forecast_table.parquet` (batter, season, k, w, θ̂, `r_final` center +
  50/80/90 lo/hi, realized `r_final`, benchmarks), `metrics.json` (gates, by-band tables, paired
  bootstrap), `figures/`, `NOTES.md`.
- **Headline figure — the fan chart:** an example player's xwOBA-to-date with the final-line
  forecast fan narrowing across the season toward his realized number. *That is the product.*

## 12. Limitations & risks

1. **Four-season window.** Multi-year is testable on 2023/24/25 forecasts (1–3 prior seasons);
   pre-2022 history is invisible. Enough to prove the lever, not a production projection depth.
2. **Aging needs external data** and is deferred to rung (c); the base lever is career pooling.
3. **Thin forward-bootstrap** at low *k* — the player's own value multiset is sparse early; the
   league-shaped reference and analytic fallback cover this, but the asymmetry is less trustworthy
   there. Name it in the NOTES.
4. **Live `D_rest` is projected, not modeled** in v1 (§5); validation uses realized `D_rest`.
5. **Gaussian latent tails on a bounded stat** (xwOBA ≥ 0). Harmless mid-distribution, wrong in the
   extreme low-PA tail — inherited from Level 2.
6. **Selection.** Players benched mid-season (few remaining PAs) are eligible at fewer cutpoints;
   the eligible set skews toward regulars at high *k*. Report eligibility counts per band.
7. **EB ignores hyperparameter uncertainty** — small at ~2,600 player-seasons; the Full-Bayes
   stretch goal quantifies it.

## 13. Testing

- **Unit:** the final-line blend identity `r_final = (1−w)r_obs + w·r_rest` reproduces a directly
  computed full-season rate to machine precision; `bootstrap_S` reuse matches `talent2`; closed-form
  posterior matches a brute-force Gaussian conditioning on a toy player.
- **Regression:** G5 (strip history → Level 2) matches `results/talent2` within tolerance.
- **Leakage:** assert the forecast for (i, t, k) touches no PA with `game_date` after the cutpoint
  and no season `> t`; a digest over the conditioning set travels in `metrics.json`.
- **Calibration self-check:** on synthetic players drawn from the fitted generative model, nominal
  coverage must hold (validates the machinery independent of real-data confounds).

## 14. Open details deferred to implementation planning

- **Exact causal `μ_t` estimator** (§4.1): league xwOBA over all PAs through the cutpoint's calendar
  date (`game_date`) vs. the league mean over all players' first-*k* PAs. Within-season league drift
  is small, so either is defensible; pick one and state it. Note: for the **G5 regression test only**
  (§9), `μ_t` is instead Level 2's full-season per-season league mean, so the reduced model matches
  `talent2` exactly.
- Rung (b) peripheral composition without double-shrinkage (§4.2).
- Exact aging-curve parameterization and the birthdate source (rung c).
- Forward-bootstrap reference distribution for thin samples (career vs league-shaped) (§5).
- Min-remaining eligibility threshold and cutpoint set finalization (§7).
- Whether AR(1) is identifiable enough at 4 seasons to include in (c).
