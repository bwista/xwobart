# Rest-of-season xwOBA forecast — shrink toward a hitter's own career, not just this season

Date: 2026-07-21 · No BART re-fit. Reproduce: `.venv/bin/python scripts/run_talent3.py`
→ `results/talent3/{figures/, forecast_table.parquet, metrics.json, leakage_digest.json}`.

Design: `docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-forecast-design.md` ·
Plan: `docs/superpowers/plans/2026-07-20-xwobart-rest-of-season-forecast.md`. Rung (a) of the
model (spec §6): career pooling + iid season drift, xwOBA-only, no aging.

## Goal

Stand at a mid-season cutpoint — a hitter has seen his first *k* plate appearances — and
forecast his **final full-season xwOBA**: a descriptor `θ̂` (his current-season talent) plus a
calibrated range that narrows as the season plays out. Phase 1 and Level 2 (`results/talent`,
`results/talent2`) each estimate a *single season* in isolation; this stage adds the one lever
they were missing — **a hitter's own completed prior seasons** — and asks whether that history
actually improves the forecast, honestly scored against naive, league, Marcel and single-season
baselines.

## Construction (rung a)

The final line is a **known-weight blend** of what already happened and what hasn't:

```
r_final = (1 − w)·r_obs + w·r_rest,      w = D_rest / (D_obs + D_rest)
```

`r_obs` is locked in; only `r_rest` (the rest-of-season rate) is uncertain, so `r_rest` is the
only thing the model has to forecast — the final-line product is that forecast down-weighted by
*w*. (This is also why the Stage-3 BART surface-uncertainty term stays shelved: once xwOBA
values are realized numbers, no redraw of the surface changes what already happened — the
remaining uncertainty is genuine future-PA sampling noise, not surface posterior uncertainty. It
cancels out of the error term; see spec §1.)

**Latent talent**, across a player's seasons:

```
θ_{i,t} = μ_t + η_i + u_{i,t}
```

- `μ_t` — the season's league environment, estimated **causally** from the first-*k* PAs of
  every player that season (never a held-out MLE; leaks nothing about the rest of season *t*).
- `η_i ~ N(0, σ_η²)` — a **career random intercept**: the player's own norm across his completed
  prior seasons. This is what carries multi-year history and is the whole lever this stage adds.
- `u_{i,t} ~ N(0, σ_u²)` iid — season-to-season deviation left after the career level. No aging
  curve and no AR(1) on `u` yet — both are rung (c), xwOBA-only and no peripherals is rung (a);
  see "Next levers."

**Measurement.** Each season's rate (completed prior seasons: full-season; the target season at
the cutpoint: first-*k* PAs) is a noisy read `z = θ + ε`, `ε ~ N(0, S)`, with `S` **reused
verbatim from `talent2.bootstrap_S`** (xwOBA-only: NaN peripherals in, take `S[0,0]`, floored at
`FLOOR_SD_PER_PA²/n`).

**Fitting.** Two different regimes for two different kinds of parameter:
- `μ_t` — causal, per season, per cutpoint, never held out (a per-season mean can't be formed
  for a fully-held-out season, which is exactly why LOSO doesn't apply to it).
- `(σ_η², σ_u²)` — marginal-MLE (L-BFGS), **leave-one-season-out**: to forecast season *t*, fit
  on the other three seasons' full-season measurements only (denom ≥ 100). Fitted values are
  tight across seasons: `σ_η ≈ 0.029` in all four LOSO fits, `σ_u` 0.0119–0.0139.

**Posterior.** Given `φ = (σ_η², σ_u²)`, closed-form Gaussian conditioning (a small per-player
Kalman solve) on the player's completed prior seasons plus his first-*k* current PAs yields
`θ_{i,t} | data ~ N(θ̂, V)` — no MCMC per cutpoint, which is what makes sweeping thousands of
(player, cutpoint) pairs tractable.

**Range.** Forward-bootstrap over the player's own causal value multiset (prior-season PAs +
first-*k* current PAs, reusing the `results/player_ci` idea): draw `θ^(b) ~ N(θ̂, V)`, resample
*m* (value, denom) pairs with replacement, additively recenter to mean `θ^(b)` (preserves the
boom/bust asymmetry — an additive shift moves location without flattening shape), blend by *w*.
Quantiles of `{r_final^(b)}` at 50/80/90% are the reported range.

**Scale.** 7,493 forecasts over **1,945 unique (batter, season) pairs** (778 distinct batters,
seasons 2022–2025). Cutpoints `k ∈ {50, 100, 150, 200, 300}`; a (player, *k*) pair is eligible
only with a real rest of season (`D_rest ≥ 30`, `MIN_REMAINING`). `n_prior` (completed prior
seasons in window) splits 0: 2,352 rows / 1: 2,213 / 2: 1,748 / 3: 1,180 — the four-season window
caps history at 1–3 priors (limitation 1). The causal leakage guard (`assert_causal`) passed on
**all 7,493 forecasts**, checking 4,674,004 conditioning rows for any PA that post-dates its
cutpoint or belongs to a later season.

## Gate panel

Pooled RMSE of the final-line forecast, model vs. the five benchmarks (naive, league-shrunk,
Marcel, single-season Level 2, Savant-to-date):

| model | naive | league | marcel | single-season L2 | savant |
|---|---|---|---|---|---|
| **0.02203** | 0.03438 | 0.02448 | 0.02270 | 0.02450 | 0.03438 |

(Naive and Savant-to-date are numerically **identical** in this evaluation — our per-PA values
already *are* Savant's `estimated_woba_using_speedangle`, so "Savant's xwOBA through *k*"
collapses onto "he keeps hitting as he has" by construction, not by chance.)

| gate | result | detail |
|---|---|---|
| **G1** beats naive, low PA-seen | **PASS** | `k≤100` (n=3,637): Δ(naive−model) **+0.01786**, 95% CI [+0.01666, +0.01920] |
| **G2** beats/ties single-season L2 | **PASS (beat)** | all rows (n=7,493): Δ **+0.00246**, CI [+0.00169, +0.00335] — **CI excludes zero: the multi-year lever pays** |
| **G3** beats/ties Marcel | **PASS (beat)** | all rows (n=7,493): Δ **+0.00067**, CI [+0.00013, +0.00117] |
| **G4** calibration (±5pp) | **FAIL** | max \|coverage−level\| = **0.0727** |
| **G5** reduces to Phase 1 | **PASS** | max\|θ−xwoba_talent\| = **5.6e-17** over 2,636 rows (η stripped, full-season cutpoint, Phase-1's own per-season μ) |

**4 of 5 gates pass.** The headline result is G2: history is not free — it beats a model that
sees only the current season, with a bootstrap CI that excludes zero. G5 is the free regression
test: strip the career term and evaluate at the full-season cutpoint, and the hierarchy's output
matches Phase 1 to machine precision, so the multi-season machinery is provably not doing
something *different* from Phase 1/Level 2 at the point where it should agree exactly with them.
**G4 is a real failure**, not a rounding call — see below.

**Coverage by *k*-band** (share of rows whose realized final xwOBA falls inside the interval):

| level | k=50 | k=100 | k=150 | k=200 | k=300 |
|---|---|---|---|---|---|
| 0.50 | 0.495 | 0.486 | 0.479 | 0.470 | **0.427** |
| 0.80 | 0.780 | 0.769 | 0.760 | 0.760 | 0.769 |
| 0.90 | 0.875 | 0.872 | 0.868 | 0.859 | 0.869 |

The 50% and 80% intervals run **narrow** (below nominal) everywhere, and the 50% gap widens as
*k* grows — the opposite of what you'd want from "more information should mean better-calibrated
uncertainty." The 90% interval stays within ±5pp at every band. See limitation 8.

## The fan chart

`figures/fan_chart_examples.png` is the product: four hitters' final-line forecast fans across
all five cutpoints, narrowing toward a horizontal line at their realized final xwOBA. The four
were picked to span the shapes the forecast has to handle — a clear above-average hitter (Aaron
Judge, 2024), a near-league-average one (Trea Turner, 2024), a hot start that faded (Bryan
Reynolds, 2023) and a cold start that surged (Willson Contreras, 2025) — not cherry-picked for a
flattering result: Judge's panel is included specifically because it **misses**, staying
conservative relative to his actual .486 line even at `k=300` (his 2024 outproduced his own
2022–23 history, which a no-aging career-intercept model structurally cannot see coming). That
panel is the G4 finding made visible in a single picture, not hidden by it.

## Limitations

1. **Four-season window.** Multi-year is testable on 1–3 prior seasons (2022 is invisible as
   history); enough to prove the lever (G2), not a production projection depth.
2. **No aging yet.** `g(age; β)` is additive in the spec but deferred to rung (c) — needs
   external birthdates (Chadwick/KIT register), not in the slim Statcast cache.
3. **Thin forward-bootstrap asymmetry at low *k*.** A player's own causal value multiset is
   sparse early in the season, so the boom/bust asymmetry the forward bootstrap is built to
   preserve is itself less trustworthy exactly where it matters most.
4. **Live `D_rest` is projected, not modeled.** Validation here uses the *realized* remaining
   denom (known ex-post, since the rate is the object under test). A live forecast needs
   games-remaining × PA-rate, which is a secondary source of error not modeled in v1.
5. **Gaussian latent tails on a bounded stat.** xwOBA ≥ 0; the Gaussian posterior puts mass where
   the stat can't go. Inherited unchanged from Phase 1 / Level 2 — harmless mid-distribution,
   wrong in the extreme low-*k* tail.
6. **Selection.** A player benched mid-season has fewer remaining PAs and drops out of
   eligibility at higher *k*, so the eligible set skews toward regulars as *k* grows. Eligible
   counts by *k*: 50→**1,945**, 100→**1,692**, 150→**1,498**, 200→**1,326**, 300→**1,032**.
7. **EB ignores hyperparameter uncertainty.** `(σ_η², σ_u²)` are point estimates from ~1,370
   players per LOSO fit; the full-Bayes robustness check (propagate hyperparameter uncertainty
   into every interval) is the spec's stretch goal, not run here.
8. **Calibration (G4) — described accurately, not spun.** The 50%/80% central intervals run
   ~5–7pp narrow, and the 50% gap gets *worse* at high *k* (0.073 undercoverage at `k=300` vs.
   0.005 at `k=50`) — short remaining runway, not long-tail uncertainty, is where the interval is
   most wrong. The 90% band holds within ±5pp throughout. The likely mechanism is the predictive
   spread being understated when little of the season is left to play out (`m` shrinks, so the
   forward-bootstrap's future-sampling term shrinks with it, while whatever the model is missing
   does not). A secondary, smaller effect worth naming: the first-*k* causal `μ_t` runs
   systematically **below** the full-season `μ_t` early in the season (measured directly:
   `k=50` league mean is 0.008–0.014 low across the four seasons, ~0.01 on average, closing to
   ~0.004–0.005 by `k=300` — plausibly the real April cold-weather offensive dip, not noise) —
   this is a low-*k* centering effect, not the driver of the high-*k* coverage gap, and the two
   should not be conflated into one story. The next lever is recalibration (variance handling at
   small *m*) and rung-b peripheral sharpening — not a single named cause.

## Next levers

- **Rung (b): peripherals.** Extend the within-season measurement to Level 2's joint MVN over
  (xwOBA, avg EV, barrel rate), so the peripheral-informed measurement feeds the hierarchy
  instead of a bare xwOBA read. Low-PA sharpening; the double-shrinkage composition (peripheral
  shrink at the measurement layer, career shrink at the hierarchy layer) is open implementation
  detail per spec §4.2.
- **Rung (c): aging + AR(1) drift.** A shared aging curve `g(age; β)` (needs external
  birthdates) and `u_{i,t} = ρ·u_{i,t−1} + e` so last season informs this one beyond the career
  mean. Second-order gains on top of (a) and (b).
