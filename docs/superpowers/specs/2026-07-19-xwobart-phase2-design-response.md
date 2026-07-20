# Phase 2 design recommendation (second-opinion review)

> Response to [2026-07-19-xwobart-phase2-design-prompt.md](2026-07-19-xwobart-phase2-design-prompt.md).
> Written 2026-07-19. Design recommendation only — no code. Takes the prompt's numbers as given
> (ELPD anchor −80107; Phase 1 r=0.489 @ PA≥100 / 0.467 @ PA≥30; findings 1–3).

## Verdict

Build Phase 2 as **two modular layers**, not one joint fit:

1. **Level 1 (surface — one expensive refit):** the v0 BART categorical model with spray angle and
   batter handedness added (3 → 5 features), and **no batter terms of any kind**. This is the only
   change on the table that can move holdout ELPD.
2. **Level 2 (talent — cheap, iterable in minutes):** a player-season **joint measurement model** in
   which the model-xwOBA rollup and the fast-stabilizing peripherals (avg EV, barrel rate) are
   treated as noisy measurements of correlated latent talents, with partial pooling across batters
   and per-season league means. This is feature set B, upgraded from "peripherals as a regression
   prior mean" to a joint shrinkage that is structurally immune to the shared-sampling-noise trap
   (risk #1 below — the biggest hole in the plan as written).

Propagate the surface's posterior draws through the Level-2 rollup so each player's posterior
carries **both** sampling and model-surface uncertainty. Do not put a batter hierarchy inside the
event-level BART fit: findings 1–2 are the empirical demonstration that there is nothing for it to
estimate, and the softmax-invariance workarounds buy ~1.5k–7.5k parameters that partial pooling will
(correctly) shrink to ≈0, at real OOM and mixing cost.

This is not a retreat from "one coherent hierarchical model." It is **cut (modularized) inference**,
and the cut is the defensible kind: the surface is identified by 360k events and the player layer
can contribute essentially nothing back to it (finding 1 is that fact in miniature), so severing
feedback loses nothing measurable and buys a Level-2 model you can refit in minutes instead of
half-hours. The composed object — surface posterior × measurement model × talent prior — is one
generative model for the estimand, and the per-player posteriors it yields are coherent.

## The load-bearing decomposition (why no event-level batter effect)

Define the estimand precisely: player-season contact talent is

    θ_i = E_{x ~ P_i}[ v(x) ]

where v(x) is the surface (expected wOBA value of a ball with features x) and P_i is player i's
distribution over contact features (EV, LA, spray, …). Between-player variance in θ has two
channels:

- **Which balls they hit** (P_i): essentially all of the talent signal.
- **Outcome-given-x deviations** (what an event-level batter effect models): measured at
  year-to-year r ≈ 0.12 and worth +0.0008 in next-season r (finding 2). Noise, for practical
  purposes — and what persistence exists is substantially *park* (half of every player's games),
  which argues for a park feature someday, not batter effects.

The event-level model conditions on x, so it can only ever see the second channel. A batter
hierarchy inside the BART fit pools the wrong quantity. The pooling you actually want — a low-PA
hitter pulled toward the crowd — is pooling of θ_i, a functional of P_i, and that lives at the
player-season level by construction. This one observation settles Q1–Q3.

## Answers to the five questions

### 1. Both — but they belong to different layers, so "one fit vs sequenced" is a false choice

Spray+handedness (A) is a property of the outcome surface; the peripheral prior (B) is a property of
the player-level rollup. Neither belongs inside the other's layer.

- **Ship B first, on the v0 surface.** No data rebuild, minutes per fit, and it upgrades Phase 1 in
  place (same inputs, better prior). You get the low-PA result and calibrated intervals before the
  cache rebuild finishes.
- **Then refit the surface once with A** and re-run Level 2 on the new rollups (a re-run, not a
  redesign).

ROI if forced to rank: B is higher for the stated talent/low-PA goal (finding 3 is your own
evidence); A is the only path to beating the ELPD anchor, and it should also clean up the
sprint-speed story (its payoff should migrate onto pulled grounders once the model can see pull).
The goals require both; the costs are wildly asymmetric, so sequencing is free. Do **not** add
batter effects to the BART fit "while you're in there" — that is the one dominated option on the
table.

### 2. Batter-effect parameterization: none at event level; if ever forced, the scalar value-axis projection

Recommended: no event-level batter effect (see decomposition above).

If you later want one — e.g., to measure how much residual park/spray signal remains after A — use
your option (b), formalized as a rank-1 projection onto the value axis:

    mu[:, e] += b_{i(e)} · c,   c = (w − mean(w)) / ‖w − mean(w)‖,   b_i ~ Normal(0, τ_b)

with w the 5-vector of linear weights. This is softmax-safe (c ⟂ the invariant direction **1**),
costs 1 parameter per batter, and b_i reads directly as "extra wOBA per ball beyond contact
quality." The K-vector option (a) is ~6–7.5k parameters chasing an r=0.12 signal — pooling will
crush it to zero and you'll have paid sampler cost to learn that. Option (c), ordered/monotone
structure, is subsumed by the c-projection: value order is the only meaningful ordering here, and
geometry (2B vs HR differ by direction, not "more") rules out ordinal likelihoods on the classes.

If you ever run both an event-level b_i and Level-2 pooling, they compete for the same residual —
keep the channel at Level 2 only.

### 3. Two-level is not merely cheaper — it's the right architecture. But fix the prior-mean construction

Embedding player aggregates as a prior on event-level effects inherits all the costs (OOM, mixing,
softmax gymnastics) to hierarchically pool a quantity you've shown is noise. The two-level model
pools the right quantity, refits in minutes, and — with surface draws propagated — delivers the same
coherent interval. Also: do not add peripherals as BART **features**; that leaks player identity
into the surface, double-counts once Level 2 shrinks, and muddles the "surface = physics of contact"
reading that makes the design interpretable.

One material upgrade to the Set-B sketch. "Shrink toward what peripherals predict" implies
regressing xwOBA on peripherals to build the prior mean. **That regression is contaminated:** the
peripherals and the xwOBA rollup are computed from the *same balls*, so their sampling errors are
positively correlated — a lucky handful of barrels raises both. A prior-mean regression fit on raw
player-seasons partly fits shared noise: β inflates, τ deflates, and low-PA "gains" appear that are
partly fictitious. Model the vector jointly instead:

- **Observed** per batter-season: z_i = (model-xwOBA_i, avgEV_i, barrel%_i)
- **Measurement:** z_i ~ MVN( (θ_i, ξ_i), S_i ), with S_i = Σ̂_within / n_i from the pooled league
  per-ball covariance of (v(x), EV, barrel indicator); add the between-surface-draw variance of the
  rollup to the xwOBA diagonal entry. S_i's off-diagonals *are* the shared noise, carried
  explicitly, so the posterior cannot double-count it.
- **Talent prior:** (θ_i, ξ_i) ~ MVN( μ_{t(i)}, Σ_talent ) — per-season league means μ_t, LKJ(2) on
  the talent correlation, HalfNormal scales (θ scale ≈ 0.03 wOBA is the right order).

E[θ_i | z_i] then does exactly what you want with no direction-of-regression choice: at low PA the
xwOBA entry of S_i is huge while the peripheral entries are small (they stabilize fast), so the
posterior leans on peripherals through Σ_talent; at high PA the xwOBA measurement dominates. Drop
the peripheral rows and it reduces to Phase 1 — a free regression test. ~2,600 rows, full Bayes in
PyMC, minutes. Fit hyperparameters on 2022–24 season pairs only; evaluate on 2024→25.

### 4. Success criterion: a pre-registered composite, matched to layers

- **Surface (A):** holdout ELPD vs −80107, identical train-subsample protocol and holdout as v0.
  Expect an unambiguous beat — spray is enormously informative for hit type given EV/LA. Treat a
  *marginal* beat as a pipeline red flag, not a success.
- **Talent (B):** next-season actual wOBA, **RMSE primary** (r secondary — r is affine-invariant, so
  it cannot see better shrinkage within a band; Phase 1 already taught this lesson), by PA band
  (30–100, 100+), pooled across 22→23, 23→24, 24→25, paired bootstrap over players. Power warning:
  n≈113 per band-pair means per-cell r differences under ~0.05–0.08 are unresolvable — which also
  means finding 3's gap (0.244 vs 0.179) is directionally sound but soft in magnitude; expect the
  realized gain to be smaller than that headline.
- **Intervals (the actual Phase-2 deliverable):** empirical coverage of next-season wOBA at
  50/80/90% within ±5pp in every PA band, plus sharpness (mean width) at fixed coverage. Predictive
  variance must include: θ posterior variance + talent drift + next-season sampling variance + the
  outcome-beyond-contact gap variance (mean contribution ≈ 0.12 × observed gap ≈ ignorable; the
  variance is not). Phase 1 structurally cannot produce PA-sensitive calibrated intervals (Task A:
  flat-in-PA widths, tail compression); Phase 2 succeeding here while merely *tying* Phase 1 at high
  PA is success, not failure.

Keep the K/BB splice identical across Phase 1/Phase 2 comparisons so deltas isolate the talent
layer.

### 5. What's missing / over-built

Missing, ranked by expected value:

1. **Multi-season pooling + age at Level 2.** θ_{i,t} = η_i + β_age·f(age_{i,t}) + ε_{i,t} (or
   AR(1)). Minutes to fit, lets 2022+2023 inform 2024 predictions, and is plausibly the largest
   remaining accuracy lever on next-season wOBA — bigger than anything in A or B. Label it clearly
   in comparisons (it uses more input than Savant's single-season stat).
2. **The K/BB channel.** wOBA ≠ wOBAcon. K% and BB% stabilize faster than anything contact-side; if
   the reconstruction splices *raw* K/BB rates, shrinking them is cheap accuracy on the headline
   metric. (If already shrunk, ignore.)
3. **Low-PA selection — a bonus argument for B:** low-PA player-seasons are not random league draws
   (call-ups, platoon bats, injuries), so a league-mean prior is mildly wrong for them. The
   peripheral-informed prior partially self-corrects: a 60-BBE call-up with a 92 mph average EV
   gets a prior that reflects it.
4. **Per-season league means for the peripherals too** (the ball changes; EV environment drifts).
5. **Spray's descriptive-vs-predictive tension** — see risk 2.
6. Housekeeping: `stand` is per-event (switch hitters); quantify hc_x/hc_y missingness (if small,
   impute league conditional mean given EV×LA and add a missing flag); verify the spray-angle sign
   convention empirically (known pull hitters must cluster on the same side for both hands). Common
   transform: φ_raw = atan2(hc_x − 125.42, 198.27 − hc_y) in degrees, mirrored by stand to make it
   pull-relative; keep stand as a separate feature so BART can recover raw direction (park
   asymmetries act on raw direction; batter skill acts pull-relative).

Over-built: the K-vector batter effect; any joint event-level refit with a batter hierarchy; park
effects (real — especially Coors — but they change the estimand away from Savant parity; defer, and
note park as the main interpretable residual channel); reading n=113 correlation differences as
precise.

## Recommended spec (summary)

**Level 1 — surface.** Same machinery as v0 (pymc-bart categorical, mu shape (5, n), p = softmax,
Categorical likelihood, per-event value = Σ_k w_k p_k), same train-subsample protocol and holdout
for anchor comparability. Features: launch_speed, launch_angle, spray_pull, stand, sprint_speed. No
batter terms, no player aggregates. Persist per-event values v^(s)(x_e) for ~100–200 thinned draws
(float32: 360k × 200 ≈ 290 MB — no OOM exposure; the fit is v0's size plus two columns).
Diagnostics: PDP/importance for spray (HR band in LA×spray), and confirm sprint speed's contribution
migrates toward pulled grounders.

**Level 2 — talent.** The MVN measurement model of §3, full Bayes. Ablation ladder:
L2a xwOBA-only (≡ Phase 1; regression-test against 0.489/0.467) → L2b + peripherals → L2c +
surface-draw variance (cheap: add to S_i; robustness check: refit Level 2 per surface draw and mix
posteriors) → L2d + multi-season/age. Score every rung on the §4 composite.

**Rollup choice under spray (A/B).** With spray in the surface, compute two rollups:
spray-conditioned (right for ELPD and same-season description) and spray-marginalized (replace
v(x_e) with its league-average over spray given EV×LA×stand — a few lines, no refit). Feed both into
Level 2 and let next-season RMSE pick the talent input.

## Risks

**1. Shared sampling noise at Level 2 — the single biggest risk.** The failure mode is silent and
flattering: a naive peripheral-prior regression manufactures exactly the low-PA improvement you are
hoping to find, in-sample, out of correlated noise. Guards: (i) the joint MVN with explicit
off-diagonal S_i handles it by construction; (ii) validate strictly cross-season (hypers on 22–24,
evaluate 24→25); (iii) diagnostic — refit with S_i off-diagonals zeroed: if the apparent low-PA gain
*jumps*, the jump is the artifact, and any implementation whose gains depend on ignoring that
covariance is fitting noise.

**2. Spray's descriptive/predictive inversion.** Conditioning player rollups on per-ball direction
credits ball-by-ball spray luck; public spray-adjusted xwOBA variants generally describe the same
season better without predicting the next one better. ELPD improves regardless (conditional density
is the right use of spray). The rollup A/B is cheap insurance; don't be surprised if the
marginalized rollup wins for talent while the conditioned one wins for description. Both are
legitimate products — label them.

**3. Practical.** pymc-bart categorical machinery is unchanged (v0's 0.12 gotchas carry over); no
new sampler risk because no per-batter parameters enter the BART graph. Watch hc missingness and the
rare 3B class (spray should *help* it — triples are geometrically concentrated). Surface-uncertainty
caveat: between-draw variance is exactly right for per-player intervals, but surface errors are
correlated *across* players in the same feature region — don't reuse these intervals for
league-aggregate claims without the per-draw refit variant.

**Expectations, stated honestly:** ELPD should beat −80107 decisively or something is wrong in the
pipeline. Next-season accuracy gains will be real but modest and concentrated below ~100 PA; high-PA
results should tie Phase 1/Savant. The genuinely new deliverable is the interval: one pipeline whose
per-player posteriors narrow with PA, calibrate at nominal coverage, and carry surface uncertainty —
none of which Phase 1 can do. Judge the phase on that.
