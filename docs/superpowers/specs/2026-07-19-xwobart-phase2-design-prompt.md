# Phase 2 design prompt (for a higher-effort second opinion)

> Self-contained prompt to paste into a higher-effort model for a design recommendation on
> Phase 2 (event-level hierarchical player model + accuracy features). Written 2026-07-19
> during the Phase-2 brainstorm, after cheap on-disk tests overturned the roadmap's
> "two-stage" variant. Carries all context and numbers — no repo access assumed.
> The open decision: add spray+handedness, a peripheral-informed batter prior, or both;
> plus the batter-effect parameterization and success criterion.

---

You are a senior Bayesian sports-modeling statistician. I need a design recommendation
(not code) for the next phase of a baseball xwOBA model. Please pressure-test my plan and
recommend a concrete model specification, with reasoning and explicit risks.

## Project context

"xwobart" reproduces MLB's public xwOBA (expected weighted on-base average — a contact-quality
hitting metric) with a Bayesian model that also yields uncertainty. Stack: PyMC + pymc-bart 0.12
(BART = Bayesian Additive Regression Trees). Data: Statcast, ~360k batted-ball events over 2022–24
(train), 2025 holdout, ~2,600 batter-seasons.

- v0 model: a BART *categorical* model predicting one of 5 outcome classes (out / 1B / 2B / 3B / HR)
  per batted ball from 3 per-event features — launch_speed (exit velocity), launch_angle, and the
  batter's sprint_speed — via mu = BART(features), shape (K=5 classes, n events); p = softmax(mu)
  over classes; Categorical likelihood. Per-event expected wOBA = sum_k(linear_weight_k * p_k).
  Rolled up to a player-season xwOBA. Result: at parity with public Savant xwOBA (player-season
  r≈0.96; predicting NEXT season's actual wOBA, r=0.481 vs Savant 0.487, naive prior-year wOBA 0.39).
  The 3-feature model is at the public metric's information ceiling.
- Phase 1 (no model change): empirical-Bayes shrinkage of each batter-season's raw xwOBA toward the
  season league mean, reliability = tau^2/(tau^2+SE^2). Predicting next-season actual wOBA:
  r=0.489 at PA>=100, r=0.467 at PA>=30 (beats raw xwOBA; ties/edges Savant). Honest interval that
  narrows with plate appearances (PA), but it's estimation-only (ignores model surface uncertainty)
  and the prior is the flat league mean.

## Phase 2 goal (what I'm designing now)

One coherent hierarchical model that estimates each hitter's true-talent xwOBA AND its uncertainty
in a single fit, with partial pooling across batters (a low-PA hitter is pulled toward the crowd).
Target: coherent per-player posteriors that fold BOTH sample-size and model-surface uncertainty
together, and — ideally — beat Phase 1 on next-season wOBA, especially at low PA. The fixed
accuracy anchor to beat is holdout ELPD (log predictive density) = -80107.

## What I empirically established (cheap tests, no re-fit, on data already computed)

1. Naive "two-stage" idea (shrink raw xwOBA toward the model's own contact-based xwOBA over the
   player's same batted balls) is a STRUCTURAL no-op: the between-player residual variance of
   (raw - model_xwoba) is ~0 (tau_resid^2 ≈ 1e-4). Reason: both are xwOBA over the SAME balls, so
   their difference is just sampling noise + a tiny sprint adjustment. It LOSES to Phase 1.
2. A batter "outcome-beyond-contact" effect (actual wOBA minus xwOBA — i.e., how much a hitter
   beats their expected) persists year-to-year at only corr≈0.12. Adding it to contact xwOBA moves
   next-season prediction from r=0.4840 to 0.4848 (negligible). So a batter random intercept, by
   itself, will not improve point accuracy — partial pooling shrinks most batter effects to ~0.
3. Fast-stabilizing peripherals DO carry real low-PA signal. Predicting next-season wOBA from
   season-T stats at PA_T 30–100 (n=113): raw xwOBA r=0.179, average exit velocity r=0.227,
   barrel rate r=0.244. At high PA, raw xwOBA wins (as expected). So peripherals are best used as a
   PRIOR that dominates when PA is low.

## Key modeling subtlety I discovered

A softmax categorical model is shift-invariant per event: adding a SCALAR batter intercept to all K
class-logits of an event cancels out entirely. So a batter effect must be either (a) class-specific
(a K-vector per batter, ~5 x 1,500 batters), or (b) a scalar "productivity" effect projected onto
the wOBA-value axis (1 param/batter, interpretable as "how much this hitter beats their expected
wOBA"), or (c) some monotone/ordered structure. Which parameterization do you recommend, given
finding #2 (the productivity signal is weak and low-persistence)?

## The decision I want your recommendation on

Because the batter effect alone won't move accuracy (findings 1–2), I want to add "real accuracy"
features. Two candidates, and I'm leaning toward doing BOTH:

- Feature set A — per-event spray angle (hit direction / pull vs. oppo) + batter handedness, added
  to the BART inputs (3 -> 5 features). Rationale: v0 is blind to direction, and sprint speed's
  payoff concentrates on pulled grounders / infield hits it can't currently see. This is the
  clearest path to beating the ELPD anchor. Cost: requires rebuilding the data cache to add hit
  coordinates (hc_x, hc_y) and stand.
- Feature set B — a peripheral-informed batter prior: shrink each batter's effect toward what their
  fast-stabilizing peripherals (barrel%, avg/max exit velocity, hard-hit%) predict, rather than
  toward 0. Directly targets the low-PA denoising win in finding #3. No data rebuild. These are
  player-season aggregates, so they'd enter as batter-level covariates on the effect's prior mean,
  not as per-event features.

Constraints/risks: the event-level hierarchical BART is the expensive part — on an 18 GB machine,
full-train already risks OOM (~14.5 GB in-memory posterior for mu); v0 used a 100k-event subsample
(~27 min fit). Adding thousands of batter effects and 2 more features increases cost and OOM risk.
Also, the original roadmap said do the simpler variant first — but I've shown that variant is dead.

## Please answer

1. Both feature sets in ONE hierarchical fit, or sequence them (e.g., spray+handedness first as its
   own model, then add the batter hierarchy)? Or is one of them clearly higher ROI than the other?
2. How should the batter effect be parameterized (class-specific vs. scalar productivity vs. other),
   given it carries weak, low-persistence signal but is needed for coherent partial-pooled intervals?
3. If peripherals mainly help at low PA and are player-season aggregates, is embedding them as a
   batter-effect prior in an event-level model the right architecture, or would a two-level model
   (event surface -> player-season partial pooling with a peripheral-informed prior) be cleaner and
   far cheaper while delivering the same coherent interval?
4. What's the most defensible success criterion, given a pure hierarchy likely won't beat Phase 1 on
   point accuracy — ELPD, next-season wOBA at low PA, interval calibration, or a combination?
5. Anything I'm missing, over-building, or getting statistically wrong.

Give a concrete recommended model spec (likelihood, batter-effect structure, priors, how peripherals
and spray/handedness enter, and the validation plan), plus the single biggest risk to watch.
