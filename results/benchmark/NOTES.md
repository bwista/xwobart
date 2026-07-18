# Benchmark — is v0 more accurate than Savant?

Date: 2026-07-18 · No re-fit. Reproduce: `.venv/bin/python scripts/benchmark_vs_savant.py`
→ `results/benchmark/{figures/predictive_accuracy.png, benchmark_metrics.json, pairs.parquet}`.

**Question.** "More accurate than Savant" only means something against *actual
outcomes*, not against Savant itself (agreeing with Savant more just means we copied
it). The canonical xwOBA-validity test: does year-T xwOBA predict year-(T+1) **actual
wOBA** better? Three predictors race — v0 model xwOBA_T, Savant xwOBA_T, and the naive
baseline (actual wOBA_T, which xwOBA must beat to justify existing). Actual wOBA is
rebuilt per player-season from the slim caches as Σwoba_value/Σwoba_denom. 1,058
player-pairs (100+ PA in both years) across 2022→23, 23→24, 24→25.

## Result — v0 is at **parity** with Savant (does not beat it)

| predictor | Pearson r ↑ | R² | calibrated RMSE ↓ |
|---|---|---|---|
| v0 model xwOBA | 0.481 | 0.231 | 0.0346 |
| **Savant xwOBA** | **0.487** | **0.237** | **0.0345** |
| naive (actual wOBA_T) | 0.390 | 0.152 | 0.0363 |

Paired bootstrap over player-pairs: `r_model − r_savant = −0.006`, 95% CI
**[−0.025, +0.013]**, model better in only 26% of resamples → **no significant
difference.** By season pair, Savant wins 2022→23 (0.516 vs 0.479), the model wins
2023→24 (0.487 vs 0.481) and 2024→25 (0.504 vs 0.498) — no consistent edge either way.
(Calibrated RMSE = residual std after an OLS rescale, so a scale/bias offset doesn't
distort the comparison; raw RMSE actually favors the model, 0.036 vs 0.038, only
because its mean sits closer to next-year wOBA.)

Both v0 and Savant clearly beat the naive baseline (r 0.48 vs 0.39; calibrated RMSE
0.035 vs 0.036), so **v0 is genuine xwOBA** — it predicts next-year talent better than
raw wOBA. It simply is **not more accurate than Savant.**

## Why, and what it means for the roadmap

v0 is essentially a *smoothed replication* of Savant's EV/LA surface plus a weak sprint
term (Task A showed the divergence from Savant is mostly tail-compression, not new
signal; the §9.4 speed-undercorrection test was inconclusive). With only launch_speed /
launch_angle / sprint_speed, the model is at **Savant's information ceiling** — there is
no free accuracy left to extract by tuning the 3-feature model.

Note the comparison is, if anything, *generous* to the model: year-T is an **in-sample**
season for it (its BBE surface was fit on 2022–24), and it still only ties. A clean
out-of-sample predictor-year comparison isn't possible with 2022–25 (2025 has no T+1).

**Conclusion / recommendation.** "Beat Savant before proceeding" is not achievable
within v0 — the only lever that can move accuracy past Savant is adding inputs Savant
lacks that carry real predictive signal, i.e. **v1 = spray angle + handedness** (speed's
payoff concentrates on pull-side/infield hits, which the current features can't see).
This benchmark is the yardstick v1 must clear:

> **v1 target: pooled Pearson r > 0.487 (calibrated RMSE < 0.0345) vs next-season
> actual wOBA, with a bootstrap CI on the gap that excludes 0.**

Recommended follow-up (needs one re-fit — the fitted BART trees are not persisted in
`idata.nc`): an **event-level** holdout head-to-head (model expected-value vs Savant
`estimated_woba_using_speedangle` vs actual `woba_value`, on the fully-OOS 2025 events).
Fold it into the v1 fit by saving per-event EVs so no extra fit is spent.
