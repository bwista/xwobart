# Rest-of-season forecast — rung b: spray as an early-settling peripheral

> Written 2026-07-23. Extends the **rest-of-season xwOBA forecast** spec
> (`docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-forecast-design.md`), which
> shipped **rung a** (career random intercept + iid drift, xwOBA-only) and reserved
> **rung b** for peripherals (§4.2, §6). This spec fixes rung b's open composition detail
> and names **spray/pull tendency** as a peripheral channel. No code here — spec only.

## 1. Why this, and why now

The product target is **in-season / short-term** forecasting — a hitter's rest-of-current-season
xwOBA from a mid-season cutpoint — **not** next-season prediction. Rung a proved the multi-year
lever (G2: beats single-season Level 2, CI excludes 0) but left two openings this spec addresses:

1. **Rung a is weakest exactly where the product matters most.** Its G4 calibration miss is at
   **short runway / low PA-seen** — the 50/80% intervals run narrow when a hitter has played little.
   That early window is the whole point of an in-season forecast.
2. **Spray carries real signal.** The matched-capacity surface experiment
   (`results/capacity_C_m200/`) showed the 5-feature spray surface decisively beats v0 at m_trees=200
   (paired +3,017 nats vs a 37-nat noise floor) — the Stage-3 "no improvement" was a capacity
   artifact. That result is about *describing* a batted ball; this spec spends the same insight on
   *forecasting*, via the most direct, decoupled route.

**Pull tendency** (how often / how far a hitter pulls the ball) **stabilizes within a few dozen
batted balls** — far faster than a rate built from outcomes — so it is informative about talent
precisely in the low-PA regime where rung a is thin. It is computed directly from hit coordinates
(`spray_pull`, the same signed pull-relative angle the surface uses), so this rung **does not depend
on the BART surface fit** at all.

## 2. Scope

Build rung b as specced in the parent (§4.2, §6): extend the within-season talent *measurement* from
xwOBA-only to the Level-2 joint MVN, with peripheral channels **avg EV, barrel rate, and pull
tendency**. The career-intercept hierarchy, the forward-bootstrap range, the cutpoint sweep, the
benchmarks, and the gate panel are all **unchanged and reused** from rung a. This spec changes only
how the per-season measurement `(z, S)` that feeds the hierarchy is formed.

Out of scope: the spray *surface* as a value function (the "Approach B" alternative — a rate target
makes a better surface partly cancel; heavier, weaker-motivated, considered and rejected); aging
(rung c); live `D_rest` projection (parent §5).

## 3. The signal

Per (batter, season) and per first-*k* cutpoint, a scalar **pull statistic** on that PA subset's
batted balls. Candidate definitions (pinned in the plan; both are BBE-only, like EV/barrel):

- **mean `spray_pull`** — the average signed pull angle (continuous; uses all the information), or
- **pull rate** — fraction of BBE with `spray_pull` above a pull threshold.

`spray_pull` is already mirrored by handedness so positive = pulled for both hands
(`src/prep.py:_spray_cols`, sign-QC'd in Stage 2). The plan picks one definition by its incremental
signal (§5) and stabilization speed.

## 4. The model — a variance-reducing, non-shrinking measurement layer

### 4.1 What rung b must produce

The hierarchy (parent §4.1) needs, for each season *s*, one Gaussian likelihood term for that
season's talent: `z'_{i,s} ~ N(θ_{i,s}, S'_{i,s})`. Rung b's only job is to make `(z', S')` a
**sharper** summary of the season's information about xwOBA talent than the xwOBA-only `(z, S)` of
rung a — by borrowing the correlated, faster-settling peripheral channels.

### 4.2 The double-shrinkage trap, and the fix

The parent (§4.2) flags the danger: `talent2`'s `mvn_posterior` shrinks the xwOBA read **toward the
population mean** (peripheral shrink), and then the career hierarchy shrinks **again** toward `μ_t`
and the player's career norm. Composing two shrinkages over-regresses.

**The trap is an artifact of splitting the inference into two *priored* stages.** The fix is to give
the season-*s* xwOBA talent `θ` a prior **exactly once** — in the hierarchy — and let the peripheral
layer contribute only a *likelihood*. Model season *s*'s four measured channels
`m = (m_xwoba, m_p)`, `m_p = (m_ev, m_barrel, m_pull)`, as a noisy read of latent season talents
`(θ, p)`, with the peripheral talents given the population prior **conditional on θ**:

```
m = (θ, p) + ε,        ε ~ N(0, S)                         [S = bootstrap_S, 4×4]
p | θ  ~  N( μ_p + β·(θ − μ_t),  Σ_{pp·θ} ),   β = Σ_pθ Σ_θθ^{-1}
```

The **conditional** prior `p(p | θ)` (not the marginal `N(μ_p, Σ_pp)`) is what carries the *talent*
correlation `Σ_θp = Cov(θ, p)` — the entire source of the borrow. `θ` itself gets **no prior in this
layer** (flat). The season's contribution to θ is its marginal likelihood, Gaussian in θ:

```
L*_s(θ)  =  ∫ p(m | θ, p) · p(p | θ) dp  =  N( θ ;  z'_s,  S'_s )
```

- **Unbiased (given the fitted `β`/`Σ`).** With θ flat here and `E[m | θ]` unit-slope in θ, `z'_s` is
  an unbiased measurement of θ — no pull toward league (the `μ_t`/`μ_p` recentering cancels from the
  bias). Peripheral centering uses the **same per-season `μ_t`** as the xwOBA channel. Under
  EB-estimated `Σ` the unbiasedness is approximate; the §4.4 synthetic test checks the exact-`β` case,
  not robustness to `Σ` misspecification (§12).
- **Borrows via talent correlation.** `m_p` enters through `β = Σ_pθ Σ_θθ^{-1}`, so the peripherals
  sharpen θ through `Σ_θp` (matching the intent), tightening `S'_s ≤ S_θθ` whenever `Σ_θp ≠ 0`.
- **One shrinkage.** The unchanged hierarchy supplies θ's single prior `H(θ)`; the player's posterior
  is `p(θ | all seasons) ∝ H(θ)·∏_s L*_s(θ)`, which is **identically the full joint inference *of this
  model*** — one shrinkage toward `μ_t` + career, peripherals folded in correctly. This is why §2
  holds: the measurement layer emits `(z'_s, S'_s)` and the rung-a hierarchy consumes them exactly as
  it consumed `(z, S)`. **Scope boundary:** peripherals are *not* pooled across a player's seasons —
  each season's peripheral read informs only that season's θ; prior-season peripherals enter only
  indirectly, through the already-sharp prior-season `θ̂`. Since pull tendency is itself stable,
  cross-season peripheral pooling could sharpen a thin current-season read further — a deliberate
  simplification here (it keeps the composition a clean single shrinkage), logged in §11.

The earlier flat-prior-with-**marginal**-`p(p)` construction is wrong precisely because it drops
`β`: the borrow would run only through measurement-noise cross-terms and re-introduce a residual pull
toward `μ_t` — the failure the §4.4 unbiasedness test is designed to catch.

**Measurement-noise cross-terms.** `bootstrap_S` returns a full 4×4 `S` whose off-diagonals are real
(the channels share PAs). The **base rung takes `S` block-diagonal between θ and the peripherals**
(`S_θp = 0`), so the borrow is purely the talent correlation `Σ_θp` and the §4.4 reduction identity is
exact; the small `S_θp` noise-cancellation borrow is a documented refinement, not part of the base
claim.

### 4.3 Fitting

The 4×4 population covariance `Σ` and channel means are EB hyperparameters, fit by maximizing the
summed marginal log-likelihood over completed player-seasons — `talent2.py:mvn_mle` **generalized
from 3 to 4 channels**. Peripheral channels are BBE-only (as EV/barrel already are); `bootstrap_S`
returns the matching per-channel measurement noise on the relevant PA subset (parent §4.2 reuse).
Structural hyperparameters `(σ_η², σ_u²)` stay under the parent's LOSO protocol (§7).

### 4.4 Reduction guarantees (tests, not hopes)

- **Zero peripheral talent-correlation → rung a exactly.** If every talent cross-covariance `Σ_θp` is
  zero (and `S` block-diagonal per §4.2), then `β = 0`, `m_p` is uninformative about θ, and
  `(z'_s, S'_s) = (m_xwoba, S_θθ)` — rung a's measurement (a strict, checkable identity; the rung-b
  analogue of G5). The test zeroes the **talent** cross-covariance `Σ_θp`, not the measurement-noise
  cross-terms.
- **No double-shrinkage / unbiased measurement.** On synthetic players from the fitted generative
  model (talents correlated, `Σ_θp ≠ 0`), `z'_s` is unbiased for θ — its across-sim mean tracks true θ
  with no attenuation toward the league mean. This guards the exact failure mode a marginal-prior
  construction would exhibit (a residual pull toward `μ_t`).

## 5. De-risk first — a cheap incremental-signal gate (go/no-go)

Pull tendency overlaps with power, which EV and barrel already capture, so spray must add value
**beyond** them. **Before** building the 4-channel model, run a cheap check on the existing forecast
table: does early-*k* pull predict the **rest-of-season xwOBA rate** *after* controlling for early-*k*
xwOBA, EV, and barrel? (partial correlation / incremental R² of the pull term, by PA-seen band; no
model fit). **Go** if pull shows non-trivial incremental signal at low *k*; **stop and report**
"spray redundant with EV/barrel for forecasting" if it is null everywhere. This can save the whole
build and is itself a reportable result.

## 6. Validation — the existing gate panel + a spray-isolating ablation

Scored on the **same** harness as rung a (`scripts/run_talent3.py`): the same eligible
(batter, season, k) pairs (≈7,493 forecasts / 1,945 player-seasons, k ∈ {50,100,150,200,300}), the
same benchmarks (parent §8), the same paired-bootstrap-over-players, reported **by PA-seen and w
band** (never pooled-only; affine-invariance caveat, parent §9).

**Three models, to isolate spray specifically** (not just "peripherals help"):

| model | measurement channels |
|---|---|
| rung a | xwOBA only (shipped baseline) |
| rung b − pull | xwOBA + EV + barrel |
| rung b + pull | xwOBA + EV + barrel + **pull** |

**Success criteria (pre-registered):**

- **Primary (hard gate) — sharper early.** rung b (+pull) beats rung a on rest-of-season `r_final`
  **RMSE at low PA-seen** (paired-bootstrap CI excludes 0). This is the pass/fail bar.
- **Spray's marginal value (reported headline).** (rung b +pull) − (rung b −pull) with its CI; the
  honest headline is this delta, not the peripheral bundle's.
- **Calibration (reported diagnostic, not a hard gate).** 50/80/90 coverage by PA-seen and *w* for all
  three models; the specific question is whether rung b **narrows the short-runway miss** toward the
  parent's ±5pp target without breaking the 90% band. A calibration that does not improve is an
  informative outcome, not a build failure.
- **Guardrails.** the §4.4 reduction identity holds; leakage guard green (§7); pooled RMSE + by-band
  table reported for all three models and all five benchmarks.

A **negative is a real result**: if +pull does not beat −pull, we report spray as forecast-redundant
and stop — consistent with the project's habit of writing up documented negatives.

## 7. Leakage

Reuse the parent's `assert_causal` guard verbatim: the pull statistic for (i, t, k) is built from
**only the first *k* PAs** (`game_date ≤ cutpoint`), never a season > t, never the rest of t. The
conditioning-set digest travels in `metrics.json` exactly as rung a's does.

## 8. Data prerequisite — verified reachable, no rebuild

The forecaster builds its PA frame from the slim caches, and `src/data.py:KEEP_COLUMNS` **already
carries `hc_x`, `hc_y`, `stand`** (added in Stage 2). So `spray_pull` derives on the forecaster's own
path via `src/prep.py:_spray_cols` — **no external data, no cache rebuild** (unlike rung c's
birthdates). The one mechanical plan-time task: carry a per-PA pull value through `build_pa_frame`,
which currently aggregates to PA level without it. Confirmed 2026-07-23.

## 9. Code layout

- `src/talent2.py` — generalize the MVN measurement (`mvn_mle`, `mvn_posterior`, `bootstrap_S`
  consumption) from a fixed 3 channels to **k channels**; keep 3-channel behaviour bit-identical
  (regression-guarded).
- `src/talent3.py` — add the rung-b measurement assembly: build the 4-channel `(m, S)` per
  season/cutpoint and emit the marginal-likelihood `(z', S')` (§4.2) that the existing hierarchy
  consumes unchanged.
- `src/prep.py` / `src/rollup.py` — per-(batter, season, first-k) pull aggregation, BBE-only.
- `scripts/run_talent3.py` — a `--rung {a,b}` selector (and a `--peripherals` toggle for the
  ablation), producing the three-model comparison and the incremental-signal pre-check (§5).
- `results/talent3/` — extend `metrics.json` with the three-model gate panel + the +pull/−pull delta;
  add a short rung-b section to `NOTES.md`; a figure showing calibration at short runway, rung a vs
  rung b.

## 10. Testing (TDD)

- **Unit.** k-channel `mvn_mle`/`mvn_posterior` reproduce the 3-channel `talent2` results exactly
  (regression); the §4.2 measurement `(z', S')` matches a brute-force Gaussian marginalization **using
  the conditional prior `p(p | θ)`** on a toy player; `(z', S') = (m_xwoba, S_θθ)` when the talent
  cross-covariance `Σ_θp` is zeroed (§4.4 identity).
- **No double-shrinkage.** synthetic-player check that `z'` is unbiased for θ (§4.4).
- **Leakage.** the digest test (§7) extends to the pull channel.
- **Regression.** rung a's shipped numbers are unchanged when `--rung a` is selected (the extension
  is strictly additive).

## 11. Limitations & risks

1. **Redundancy.** Pull may add little beyond EV/barrel; §5 catches this cheaply, and the ablation
   quantifies whatever remains.
2. **Pull-definition sensitivity.** mean-angle vs pull-rate may differ at low *k*; the plan picks by
   incremental signal and states the choice.
3. **BBE-only, thin early.** the pull read is over batted balls, sparse at very low *k*;
   `bootstrap_S` carries the honest measurement noise, so a thin read simply contributes little —
   but the asymmetry caveat of the parent (§12.3) applies to this channel too.
4. **Gaussian tails on a bounded stat** — inherited from Level 2 (parent §12.5); unchanged.
5. **Four-season window** (parent §12.1) — unchanged; the multi-year lever depth is not affected by
   the measurement rung.
6. **No cross-season peripheral pooling.** Peripherals inform only their own season's θ; a stable trait
   like pull tendency could in principle be pooled across a player's seasons to sharpen a thin
   current-season read further. Deliberately out of scope for this rung (it keeps the composition a
   clean single shrinkage, §4.2); a natural refinement if rung b pays.

## 12. Open details deferred to implementation planning

- Pull statistic definition (mean angle vs pull rate vs a robustified variant) and its BBE threshold.
- Numerically stable form of the §4.2 measurement `(z', S')` (the construction is pinned in §4.2;
  only the stable linear-algebra implementation remains).
- Whether to include the `S_θp` measurement-noise cross-terms as a refinement beyond the
  block-diagonal base (§4.2).
- Whether pull's population correlation with θ is stable enough across four seasons to trust the
  cross-channel borrow (report the fitted `Σ`).
