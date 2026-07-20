# Level-2 true talent — shrink toward what the peripherals imply, not the league mean

Date: 2026-07-19 · No BART re-fit. Reproduce: `.venv/bin/python scripts/run_talent2.py --stage full`
→ `results/talent2/{figures/, talent2_table.parquet, l2a_table.parquet, talent2_metrics.json}`.

Design: `docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md` ·
Plan: `docs/superpowers/plans/2026-07-19-xwobart-phase2-level2-talent.md`. Stage 1 of Phase 2.

**Goal.** Phase 1 (`results/talent/NOTES.md`) shrinks every hitter toward the **season league
mean**, which is the right answer only when you know nothing else about him. But you do know
something else: how hard he hits the ball. Exit velocity and barrel rate stabilize far faster
than xwOBA, and at 30–100 PA they out-predict next season better than raw xwOBA does
(barrel r 0.244, EV r 0.227, raw xwOBA r 0.179). Level 2 puts them in the **prior**, so a
rookie barreling the ball regresses toward a barrel-hitter's xwOBA instead of toward .310.
This closes Phase-1 limitation 1.

## Construction

The three stats are **jointly noisy measurements of correlated latent talents** — not a
regression of one on the others (see the shared-noise section below for why that distinction
is the whole ballgame):

- measurement: `z_i = (xwOBA, avg EV, barrel rate)_i ~ N((θ_i, ξ_i), S_i)`
- talent: `(θ_i, ξ_i) ~ N(μ_season, Σ_talent)`
- posterior: `θ̂ = μ + Σ(Σ+S_i)⁻¹(z−μ)`, `V = Σ − Σ(Σ+S_i)⁻¹Σ` — closed form, no MCMC.

1. **Per-PA measurement frame** (`build_pa_measurements`). xwOBA value exactly as Phase 1.
   EV and barrel (`launch_speed_angle == 6`) are recorded **only on tracked BBE**; the ~0.3%
   of batted balls with no tracking keep their xwOBA value but leave the peripheral
   denominators alone.
2. **Per-player measurement covariance `S_i` by bootstrap** (`bootstrap_S`). Resample the
   player's own PAs with replacement, recompute all three stats per replicate, take the
   covariance. This is the load-bearing step: it captures the **cross-covariances** exactly
   (all three stats come from the same balls) and handles the mixed denominators — xwOBA is
   per-PA, EV and barrel are per-tracked-BBE — with no analytic approximation.
   `S_i[0,0]` is floored at `(0.25)²/n` (see "the 136 rows" below).
3. **Hyperparameters by marginal MLE** (`mvn_mle`) on the stable population (PA ≥ 100,
   n = 1,830): per-season `μ` and one shared `Σ_talent`, with `Σ` parameterized by its
   Cholesky factor so it is PSD by construction. Inputs standardized per dimension.
4. **Closed-form posterior for every player-season** (2,636). Rows with fewer than 5 tracked
   BBE, missing peripherals, or a failed bootstrap fall back to the 1-D xwOBA-only model —
   i.e. exactly Phase 1's machinery. **93 rows (3.5%) take the fallback; 2,543 use all three
   dimensions.** Median reliability rises 0.65 (Phase 1) → 0.70.

### Fitted talent correlations (`Σ_talent`, unstandardized)

|  | xwOBA | avg EV | barrel rate |
|---|---|---|---|
| **xwOBA** | 1.000 | **+0.776** | **+0.712** |
| **avg EV** | +0.776 | 1.000 | +0.824 |
| **barrel rate** | +0.712 | +0.824 | 1.000 |

Talent SDs: xwOBA **0.0312**, EV **2.42 mph**, barrel rate **0.0367**. The xwOBA figure is
the sanity check that matters — Phase 1's per-season τ ranges 0.0307–0.0323, so the single
shared Σ lands mid-range and the "one Σ across seasons" simplification costs nothing.
Per-season μ (xwOBA dim) 0.3034 / 0.3126 / 0.3094 / 0.3166 for 2022–25, within 0.0016 of
Phase 1's per-season μ everywhere.

## What it does to actual hitters

Low-PA seasons where the contact quality disagrees with the results (`next` = the following
season's actual wOBA, which neither model saw):

| batter-season | PA | EV | barrel | raw | Phase 1 | **Level 2** | next |
|---|---|---|---|---|---|---|---|
| Sal Stewart 2025 (hot line, elite contact) | 57 | 95.4 | .179 | .409 | .334 | **.364** | .361 |
| Stone Garrett 2022 (average line, good contact) | 76 | 93.9 | .109 | .307 | .305 | **.337** | .420 |
| Tyler Heineman 2023 (hot line, no contact) | 42 | 77.4 | .000 | .332 | .320 | **.275** | .292 |
| César Salazar 2024 (**a miss**) | 31 | 82.9 | .000 | .347 | .318 | **.274** | .352 |

The first three are the mechanism working: Stewart's hot 57 PA is *backed* by 95.4 mph and a
17.9% barrel rate, so Level 2 declines to shrink him all the way and lands on .364 against a
.361 next season; Heineman's identical-looking hot line came off 77.4 mph contact with zero
barrels, and Level 2 calls it luck. Salazar is included because it is a **real miss** — same
signal, wrong answer, next-season .352. The peripherals are information, not a guarantee, and
n = 31 is n = 31.

`figures/peripheral_pull.png` shows the whole population: the correction fans out as PA
shrinks and is cleanly signed by barrel rate (high-barrel pulled up, low-barrel pulled down),
collapsing to zero above ~300 PA where the xwOBA sample already knows.

## Validation — does it predict next season better?

Season-T estimate vs season-(T+1) **actual** wOBA, target year PA ≥ 100. Pearson r
(calibrated RMSE in parentheses):

| population | n | **Level 2** | Phase-1 talent | raw | Savant |
|---|---|---|---|---|---|
| pooled, PA_T ≥ 100 | 1060 | **0.4908** (0.03446) | 0.4886 (0.03451) | 0.4835 (0.03462) | 0.4908 (0.03446) |
| pooled, PA_T ≥ 30 | 1173 | **0.4698** (0.03522) | 0.4669 (0.03528) | 0.4454 (0.03572) | 0.4521 (0.03558) |

**Gates: G3 (low-PA win) PASS, G4 (high-PA non-inferiority) PASS.** At PA ≥ 30 Level 2 beats
Phase 1 on both r and calibrated RMSE, and beats Savant by 0.0177. At PA ≥ 100 it gains
0.0022 over Phase 1 and now **exactly ties Savant** (0.4908 vs 0.4908) where Phase 1 trailed.

**Where the gain lives** — by PA band, the design's prediction made visible:

| PA band | n | Level 2 | Phase 1 | Δ | raw | Savant |
|---|---|---|---|---|---|---|
| [30,60) | 50 | **0.3107** | 0.2389 | **+0.0718** | 0.2088 | 0.2111 |
| [60,100) | 63 | **0.1247** | 0.0882 | +0.0365 | 0.1102 | 0.1283 |
| [100,250) | 227 | **0.2935** | 0.2769 | +0.0166 | 0.2783 | 0.2931 |
| [250,inf) | 833 | 0.5122 | 0.5122 | −0.0000 | 0.5164 | 0.5230 |

A monotone gradient that vanishes exactly where it should. Note this table means something
here that the equivalent Phase-1 table did **not**: Phase 1's per-band r was affine-invariant
to shrinkage (within a band `θ̂` is ~affine in raw, so r cannot move — see
`results/talent/NOTES.md`), which is why its per-band gaps were noise. Level 2 adds
*independent information* inside the band, so its per-band gains are real signal. They are
also small-n (50, 63) and correspondingly noisy — read the gradient, not the digits.

### Honest accounting: the gain is real in direction, not established in size

Three things cut against reading +0.0029 as a solid win, and all three are in
`talent2_metrics.json`:

1. **The paired bootstrap CI straddles zero.** 5,000 resamples of the PA ≥ 30 pairs:
   Δr mean **+0.0029, CI95 [−0.0117, +0.0176]**, better in **64.1%** of resamples. The point
   estimate favors Level 2; the sample cannot distinguish it from zero.
2. **The confirmation season reverses sign (G6).** Model choices were scored on
   *select* = 22→23 and 23→24; *confirm* = 24→25 was held out. Select: **+0.0063** (n=787).
   Confirm: **−0.0034** (n=386). The pooled +0.0029 is an average of a win and a loss, and
   nothing was tuned on confirm to fix that.
3. **The ablations don't agree with each other.** On select, full 3-D (+0.0063) beats
   barrel-only (+0.0035) beats EV-only (+0.0020), so the protocol ships full 3-D. On confirm,
   EV-only is the only positive variant (+0.0050) while full 3-D (−0.0034) and barrel-only
   (−0.0065) are negative. No variant is consistently ahead.

Note also that Δr and Δ(calibrated RMSE) are **not independent evidence**: for a fixed target
set the OLS-residual RMSE is exactly `sd(target)·√(1−r²)`, so beating on r implies beating on
calibrated RMSE, always. G3's two criteria are one criterion. (Verified to machine precision;
`_fast_scores` in the runner documents the identity.)

The fair summary: the mechanism is doing what it was designed to do, in the regime and with
the sign the design predicted, and it costs nothing at high PA — but a single season of
holdout contradicts it and the CI includes zero. It ships as an improvement in expectation,
not as a demonstrated one.

## The shared-noise tripwire (G5) — the one that mattered

**Why this is design risk #1.** The obvious way to build this — regress xwOBA on EV and barrel
rate, use the fitted value as the prior mean — is broken, and quietly so. All three stats are
computed from *the same batted balls*, so their sampling errors are correlated: a hitter who
happened to square up a few extra balls in 40 PA has an inflated xwOBA **and** an inflated EV
and barrel rate together. A regression cannot tell that shared noise from shared talent, so β
inflates, τ deflates, and the model manufactures low-PA gains out of nothing. The joint MVN is
immune by construction *only because* `S_i` carries the off-diagonals that encode exactly that
correlation.

**The test.** Refit with the off-diagonals of every `S_i` zeroed — i.e. deliberately commit the
error — and see whether the low-PA gain gets *bigger*. If it does, the gain was the artifact.

| fit | PA ≥ 30 gain vs Phase 1 |
|---|---|
| proper `S_i` (shipped) | **+0.0029** |
| off-diagonals zeroed | **−0.0101** |
| artifact gap | **−0.0130** (alarm threshold: > +0.005) |

**Clean, and emphatically so.** Zeroing the off-diagonals does not inflate the gain — it
destroys it, leaving a model *worse than Phase 1*. That is the strongest single result here:
the shared-noise modeling is load-bearing, and the +0.0029 is not correlated noise being
fitted. Had this come back positive, the low-PA win would have been withdrawn.

## The 136 rows where the variance floor binds (Phase-1 limitation 3, closed)

Phase 1 measures a sample's noisiness by how much the hitter's per-PA outcomes vary. A hitter
with **2 PA who made 2 outs** has zero variation, so Phase 1 computes se² = 0, reliability
1.0, and reports his true talent as **exactly .000** — with total confidence, off two outs.
Seven rows hit se² exactly zero; 136 fall below the floor. Trevor Rogers (a pitcher, 2 PA,
2025) is the archetype: Phase 1 .000 → Level 2 .309.

Level 2 floors the xwOBA measurement variance at `(0.25)²/n`, so no tiny sample can claim
certainty. Median PA among the 136 is 12; median move 0.0102; max move 0.3085.

**This changes the L2a regression gate, deliberately.** G1 asks whether the rebuilt machinery
reproduces Phase 1. It does — on the 2,500 rows where both models *should* agree, corr =
**0.99950** (median |diff| 0.00025, max 0.0055), and the frozen validation anchors reproduce
to 0.4885 vs 0.4886 (PA ≥ 100) and 0.4663 vs 0.4669 (PA ≥ 30). The 136 floor-bound rows are
excluded from that correlation and reported separately (`l2a.floor_fix` in the metrics,
including the all-rows corr of 0.7620 that the outliers produce) rather than folded into a
pass. Bootstrap SEs also match Phase 1's analytic SEs: corr **0.9965**, median ratio **1.002**
(G2).

## Leakage sensitivity — and a convention we are deviating on

Hyperparameters are fit on **all four seasons**, matching Phase 1's convention. The spec's
letter would fit them on the training seasons only. Refitting on 2022–24 (n_fit 1,371,
excluding 2025 measurement rows) moves the PA ≥ 30 result by **−0.0005** (0.4694 vs 0.4698) —
negligible, so the convention is kept for Phase-1 comparability. Note the exposure is
structurally small anyway: 2025 talent estimates are never a *predictor* in these races, only
a target, so 2025 can only leak through the shared Σ and the 2025 μ.

## Interval widths

Median 90% interval width, Phase 1 → Level 2 (`figures/interval_width_vs_pa.png`):

| PA band | [30,60) | [60,100) | [100,250) | [250,450) | [450,inf) |
|---|---|---|---|---|---|
| Phase 1 | 0.0869 | 0.0787 | 0.0667 | 0.0536 | 0.0466 |
| Level 2 | 0.0788 | 0.0720 | 0.0604 | 0.0498 | 0.0437 |
| change | −9.3% | −8.6% | −9.4% | −7.2% | −6.1% |

The plan expected narrowing concentrated at low PA; what the data shows is **near-uniform
6–9% narrowing at every band**. That is coherent with the model — peripherals inform θ even
when xwOBA is well measured — but see limitation 2: it is an unvalidated claim of extra
precision, and at 250+ PA the model narrows the interval by 7% while moving the point estimate
by −0.0000. Coverage validation is Stage 4's job, not this stage's.

## Limitations

1. **Single-season only.** Each player-season is estimated in isolation; a hitter's own prior
   seasons are ignored. Multi-season + age pooling at Level 2 is the flagged next lever and is
   likely worth more than the peripherals were.
2. **The interval is still estimation-only, and now narrower without being validated.** No
   BART *surface* variance is folded in (Phase-1 limitation 2 is untouched). Stage 3 persists
   per-event value draws and Stage 4 adds that variance to `S_i[0,0]` and validates 50/80/90%
   coverage by PA band. Until then the widths above are a model claim, not a measured one.
3. **Gaussian tails on a bounded stat.** xwOBA is bounded below by 0 and barrel rate lives in
   [0,1]; the MVN puts mass outside both. Harmless in the middle of the distribution, wrong in
   the extreme low-PA tail.
4. **The gain is not statistically established** — CI straddles zero, confirm season reverses.
   See "Honest accounting" above. Do not quote +0.0029 without the CI.
5. **`MIN_BBE = 5` and the 0.25 SD floor are judgement calls**, not fitted. 93 rows take the
   1-D fallback under the current threshold; neither constant was tuned, and neither was
   swept.
