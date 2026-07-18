# Task A — uncertainty sanity-check + disagreement leaderboard

Date: 2026-07-18 · No MCMC / no re-fit. Source: `results/stage_C/player_table.parquet`
(v0 frozen Stage C), 1,829 player-seasons with 100+ PA across 2022–2025.
Reproduce: `.venv/bin/python scripts/task_a_uncertainty.py`
→ `figures/`, `task_a_metrics.json`, `disagreement_top.csv`.

Purpose (per the v1 handoff): validate v0's stated core value-add — per-player
credible intervals — before investing in v1. Two checks: (1) do interval widths
behave, (2) where does the model most disagree with public Savant xwOBA.

---

## Finding 1 — the intervals are honest, but they do **not** encode sample size

The handoff expected `log(width)` vs `log(PA)` to slope ≈ **−0.5** (uncertainty
~1/√PA). It does not. The fitted slope is **+0.049** (95% CI +0.038…+0.059, R²=0.05):
essentially flat, and if anything drifting *up* with PA. Median CI width by PA bin:

| PA bin | n | median width |
|---|---|---|
| 100–150 | 220 | 0.0564 |
| 150–200 | 189 | 0.0566 |
| 200–300 | 308 | 0.0569 |
| 300–400 | 306 | 0.0574 |
| 400–500 | 288 | 0.0574 |
| 500–750 | 518 | 0.0600 |

Under a 1/√PA law the 500–750 bin should be ~2.2× *narrower* than the 100–150 bin
(~0.025); instead it is slightly *wider*. Width correlates more with the player's
value/contact profile (width vs `xwoba_mean` r=**+0.39**) than with PA (r=**+0.22**,
wrong sign for shrinkage). See `figures/interval_width_vs_pa.png`.

**Why (and it's not a bug).** These are posterior intervals over a *single shared
global* BART surface. Within a posterior draw, every event — for this player and
every other — is scored by the same sampled trees, so a player's per-draw mean
expected-value is `mean_i f_s(x_i)` where the `f_s(x_i)` are strongly positively
correlated across events. The across-draw variance of that mean therefore tends to
the *average pairwise covariance* of the surface, which is ~constant in the number
of events, not the 1/N of independent sampling. Higher-PA players are also
disproportionately good hitters sitting in higher-value, lower-density regions where
the surface is genuinely more uncertain — which is why width drifts slightly *up*
with PA rather than staying flat.

**Implication for v1.** The v0 interval faithfully answers *"how uncertain is the
fitted surface at this player's contact profile"* — it is **not** a sample-size
confidence interval and should not be sold as one. If we want "small sample → wider
interval" behavior, that needs a different construction (e.g. bootstrap resampling a
player's own BBE, or a hierarchical batter effect), not the current per-draw rollup.
Reassuringly, the intervals are the right *magnitude*: Savant's xwOBA lands inside the
model's 90% CI for **94.4%** of player-seasons — the shape is wrong, the scale is fine.

---

## Finding 2 — the model compresses the tails (clean, expected shrinkage)

Regressing `model_mean` on `savant`: slope **0.81** (<1) — the model shrinks the
extremes toward the mean. Equivalently `diff = model − savant` slopes **−0.19** on
savant (r=−0.58): the model reads ~+0.02 *high* for a 0.25-xwOBA hitter and ~−0.02
*low* for a 0.45 hitter. Overall bias is a small +0.009 (the typical player is
below-average and gets nudged up); mean |diff| is 0.013.

The `top 25 |diff|` leaderboard (`figures/disagreement_leaderboard.png`) makes it
concrete — every "model lower" case is an elite power hitter, every "model higher"
case is a weak-contact/high-groundball hitter:

| player (season) | PA | model | savant | diff | sprint | mean EV | GB rate |
|---|---|---|---|---|---|---|---|
| Santiago Espinal '25 | 320 | 0.304 | 0.254 | **+0.050** | 27.1 | 86.5 | 0.42 |
| José Abreu '24 | 120 | 0.250 | 0.201 | +0.049 | 25.3 | 87.8 | 0.52 |
| Vinny Capra '25 | 101 | 0.253 | 0.211 | +0.043 | 28.3 | 85.2 | 0.53 |
| Luis Matos '25 | 183 | 0.316 | 0.274 | +0.042 | 27.4 | 88.0 | 0.46 |
| Robinson Canó '22 | 103 | 0.277 | 0.235 | +0.041 | 24.4 | 88.2 | 0.54 |
| … | | | | | | | |
| Mike Trout '22 | 473 | 0.360 | 0.398 | −0.038 | 29.3 | 91.7 | 0.24 |
| Aaron Judge '22 | 656 | 0.430 | 0.468 | −0.038 | 27.3 | 96.0 | 0.37 |
| Aaron Judge '25 | 641 | 0.423 | 0.460 | −0.037 | 27.1 | 95.4 | 0.33 |
| Aaron Judge '24 | 682 | 0.439 | 0.480 | −0.041 | 26.8 | 96.2 | 0.31 |
| Aaron Judge '23 | 448 | 0.424 | 0.466 | −0.042 | 26.7 | 97.6 | 0.30 |

**Hand check.** The pattern is exactly what a nonparametric smoother does at the
extremes, compounded by the 5-class linear-weights structure (max class value = HR at
2.0): Aaron Judge's real xwOBA sits in the 0.46–0.48 tail and the model pulls him to
~0.42–0.44 in all four seasons; Trout likewise. At the other pole, replacement-level
contact hitters (EV 83–88, GB 0.42–0.54) get nudged up ~0.04–0.05. This is not a
speed story (the "model higher" group spans 24–29 ft/s) — it is tail compression.
The disagreements are largest-magnitude but modest (|diff| ≤ ~0.05, and the 95th
percentile of |diff| over all 1,829 is 0.028), consistent with the v0 player-season
r of 0.95.

---

## Bottom line

- **Point estimates are sound** (player r 0.95, |diff| small, tails compressed in the
  expected direction). v0's rollup is trustworthy as a central estimate.
- **The credible intervals are the honest thing but the wrong tool for "confidence
  by sample size."** They measure surface/epistemic uncertainty at a player's contact
  profile and are ~flat in PA. Don't market them as sample-size error bars.
- **For v1:** spray + handedness may sharpen the mid-range and, by giving the model a
  channel for pull-side/infield hits, could reduce some tail compression — but the
  HR-value cap in the linear-weights structure will still limit how much of Judge's
  extreme it can recover. If sample-size-aware uncertainty is a goal, that is a
  separate modeling change (bootstrap-over-BBE or a hierarchical batter term), not a
  feature addition. Recommend proceeding to Task B (spray + handedness) with these two
  properties in mind.
