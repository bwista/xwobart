# Sample-size-aware per-player xwOBA interval — "where could their xwOBA be?"

Date: 2026-07-18 · No re-fit. Reproduce: `.venv/bin/python scripts/player_ci_bootstrap.py`
→ `results/player_ci/{figures/, player_ci.parquet, ci_metrics.json}`.

**Goal.** The product is a per-player band that shows, given the N plate appearances a
hitter has put up, where their true xwOBA could plausibly sit — wide early in a season,
tightening as PA accrues. v0's model interval does **not** do this (Task A: it is ~flat
in PA — it measures BART surface uncertainty at a contact profile, not sample size).

**Construction.** Per player-season, each PA gets an xwOBA value (batted ball →
`estimated_woba_using_speedangle`, else deterministic `woba_value`); xwOBA = Σvalue/Σdenom.
We bootstrap-resample the player's PAs (B=1000) and take the 5th/95th percentiles.
Model-agnostic, no re-fit; base values are Savant's (v0 is at parity, and band *width*
is the same either way).

## It works — the band now behaves like a real confidence interval

| PA bin | **bootstrap** width | v0 model width |
|---|---|---|
| 100–150 | **0.102** | 0.056 |
| 150–200 | 0.085 | 0.057 |
| 200–300 | 0.073 | 0.057 |
| 300–400 | 0.062 | 0.057 |
| 400–500 | 0.058 | 0.057 |
| 500–750 | **0.051** | 0.060 |

- `log(width)` vs `log(PA)` slope = **−0.42** (ideal 1/√PA is −0.5; shallower only
  because higher-PA hitters are also higher-variance, which widens their band).
- Bootstrap width vs the analytic sampling SE (`sd(value)/√PA`) correlates **0.994** —
  confirming it is a genuine sampling interval.
- The band is naturally **asymmetric** (a boom-or-bust hitter has more upside room than
  a slap hitter) — real "where could it go" information a symmetric ± can't show.

## The important finding: v0's interval understates small-sample uncertainty

The two bands **cross at ~400 PA** (`figures/width_vs_pa_bootstrap_vs_model.png`):
- Below ~400 PA, v0 is **too narrow** — at 100–150 PA it reports 0.056 when the honest
  sampling band is 0.102 (**~1.8× too tight**), exactly the regime ("short term, small
  sample") this product is for.
- Above ~400 PA the two are comparable (v0 slightly wider).

So for the intended use, v0's model interval would give a false sense of precision on
exactly the players you'd most want a wide band on. See `figures/example_player_bands.png`
(Mastrobuoni 105 PA, Hilliard 152 PA: bootstrap clearly wider; Turner 538, Nimmo 660:
comparable).

## What this means for the model (honest)

The band the product needs is **model-agnostic** — it comes from resampling PAs, not from
BART. Two consequences worth stating plainly:

1. **v0's BART posterior is not the right uncertainty for this product.** Its interval is
   a *surface* interval, roughly flat in PA; it should not be shipped as a "how confident
   are we" band.
2. **The fullest, most correct interval combines two pieces** — and this is where BART
   *does* have a role:
   - **sampling** (bootstrap, dominant at low PA) — which N balls they got;
   - **surface / EV** (BART posterior, ~flat ≈ 0.056, dominant at high PA) — how well we
     know each ball's true xwOBA value.
   Combined in quadrature, `width ≈ √(sampling² + surface²)`: sampling rules the small-PA
   regime, the BART surface term rules the large-PA regime. That gives BART a clear job
   (the surface term) instead of mis-using its posterior as the whole interval.

## Recommendation

- **Ship the sampling band now** — it is the product asked for, and it's done/cheap.
- If we want the point estimate to carry the model's sprint-aware signal (and later
  spray/handedness), re-center the band on the model's xwOBA and, when we next re-fit,
  **persist per-event model EVs** so the band can be built on the model's own values and
  the surface term folded in for the combined interval.
- This reframes the roadmap: the per-player band does **not** depend on beating Savant or
  on v1 features. v1 (spray + handedness) only matters here if a sprint/pull-aware
  *center* is worth more than a Savant-based center — which, given v0↔Savant parity, is
  currently marginal.
