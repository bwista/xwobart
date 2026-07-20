# Rollup A/B — does crediting a hitter for *where* he hit it help or hurt?

Date: 2026-07-20 · No re-fit. Reproduce: `.venv/bin/python scripts/rollup_ab.py`
→ `results/rollup_ab/{rollup_ab_metrics.json, marginalized_values.parquet, figures/}`.

**The question.** Once the surface can see spray direction, a player's rollup can be built
two ways. The **conditioned** rollup values each ball at its *actual* spray angle — so a
hitter who happened to pull three balls down the line gets credit for it. The
**marginalized** rollup replaces each ball's spray with the league average over spray for
that (EV × LA × stand) cell — 9 equal-mass quantiles, no refit, just 9 extra prediction
passes — so the hitter is credited for the *contact*, not for where it happened to land.

Design risk 2 said the conditioned version would describe the season better while
predicting the next one worse, because per-ball direction is substantially luck. The race
below settles it: **season-T rollup vs season-(T+1) actual wOBA**, calibrated RMSE primary.

## Verdict: conditioning on spray HURTS, and marginalizing recovers most of it

Calibrated RMSE against next-season actual wOBA (×10⁻³, **lower is better**):

| pool | n | conditioned | marginalized | v0 (3-feature) | Savant |
|---|---|---|---|---|---|
| pooled PA ≥ 30 | 1,183 | 36.59 | 36.27 | **35.82** | **35.56** |
| pooled PA ≥ 100 | 1,072 | 35.44 | 35.07 | **34.59** | **34.45** |

By season-T PA band (PA ≥ 30 pool):

| band | n | conditioned | marginalized | v0 | Savant |
|---|---|---|---|---|---|
| 30–60 | 51 | 45.19 | 45.13 | **44.78** | 45.32 |
| 60–100 | 63 | 38.37 | 38.39 | 38.37 | **37.75** |
| 100–250 | 227 | 35.80 | 35.64 | 35.44 | **35.43** |
| 250+ | 831 | 34.90 | 34.49 | 33.89 | **33.76** |

And by season pair — the ordering is stable, not a one-year artifact:

| pair | n | conditioned | marginalized | v0 | Savant |
|---|---|---|---|---|---|
| 2022→2023 | 383 | 37.71 | 37.31 | 36.62 | **35.85** |
| 2023→2024 | 403 | 36.77 | 36.47 | 36.20 | **36.02** |
| 2024→2025 | 386 | 34.45 | 34.20 | 33.76 | **33.78** |

**Marginalized beats conditioned in 7 of the 8 band/pair splits** (the exception, 60–100 PA,
is a 0.02 tie on 63 players). Paired bootstrap of `conditioned − marginalized`, 5,000
resamples at seed 42:

| pool | mean Δ RMSE | 95% CI | conditioned better in |
|---|---|---|---|
| PA ≥ 30 | +0.000317 | [+0.000189, +0.000458] | **0 of 5,000** |
| PA ≥ 100 | +0.000375 | [+0.000225, +0.000535] | **0 of 5,000** |

Read that carefully, because the two halves say different things. The **direction** is as
resolved as a bootstrap can make it — the CI excludes zero and conditioned never once won
across 5,000 resamples. The **magnitude** is ~0.0003 wOBA, which is below the 0.001
practical bar this plan set for calling a difference meaningful at n ≈ 1,000. So: *reliably*
worse, but only *slightly* worse. Both statements are true and neither should be dropped.

Pearson r tells the same story (pooled PA ≥ 30): conditioned 0.395 < marginalized 0.413 <
v0 0.437 < Savant 0.451. Note this is **not independent evidence** — for a fixed target set
calibrated RMSE ≡ sd(target)·√(1−r²), so ranking by r and by calibrated RMSE is one
criterion, not two.

## The descriptive side, and a caveat about how to read it

Same-season correlation with Savant's public xwOBA at PA ≥ 100 (n = 1,829):

| rollup | corr vs Savant |
|---|---|
| conditioned | 0.901 |
| marginalized | 0.921 |
| v0 (3-feature) | 0.948 |

The design predicted an **inversion** — conditioned describing better while marginalized
predicts better. We do not observe one. Conditioned agrees *least* with Savant and predicts
*worst*.

But the descriptive metric here is agreement with a **spray-blind reference**. Savant's
xwOBA does not use hit location, so a spray-conditioned rollup is *mechanically* expected to
drift away from it — lower agreement is not by itself evidence of worse description. What
this table can support is narrower and still useful: conditioning moves the rollup away from
the public number *and* away from next season, so the drift is not buying predictive
information. Testing "describes the same season better" properly would need same-season
*actual* wOBA as the target, which is a different experiment.

## Verdict for Stage 4 — and why it is moot anyway

By the letter of the race, **marginalized** is the rollup to prefer: it wins at both PA
thresholds, in 7 of 8 splits, and the bootstrap never reverses. By this plan's own
practical-significance rule the ~0.0003 gap is a tie, so `rollup_ab_metrics.json` records
`stage4_talent_input: "tie"` while `pa30_lower_rmse` / `pa100_lower_rmse` both name
marginalized. Prefer marginalized; do not oversell the margin.

**In practice the choice does not arise.** Stage 3's gate E1 failed — the 5-feature spray
surface does not beat the frozen 3-feature v0 anchor (see `results/RESULTS.md`) — and both
spray rollups are beaten by v0 in *every* band and *every* season pair here. There is no
spray rollup worth promoting into the talent layer, so the Stage-4 premise as written
(push the A/B winner through Level 2) needs rethinking before it is worth building.

The one clean, transferable finding: **crediting a hitter for per-ball spray direction is
reliably counterproductive for predicting his next season.** That holds regardless of
whether the underlying surface is any good, and it is the direction the design's risk 2
predicted.

## Alignment contract for anything that reuses these arrays

`ev_draws_{tag}.npy` (200 draws × n) and `ev_marginalized_{tag}.npy` (n) are **positionally**
row-aligned with `ev_draws_keys_{tag}.parquet`, whose `row` column is the index. Train is
363,595 events, holdout 122,006 — concatenate keys and arrays in the *same* order and drop
`row` before concatenating (it restarts at 0 in each file). `rollup_ab.py` asserts this
before doing anything else; an off-by-one here is silent.

The marginalized values came from `scripts/marginalize_spray.py`, which predicts from the
**pickled** trees in a separate process (26 min, M = 9, sparse-cell rate 0.47% train /
0.56% holdout, mean |shift| vs conditioned 0.0265). That path was verified against the
in-process predictor at corr 0.99989 / mean |diff| 0.0027 before it was trusted.
