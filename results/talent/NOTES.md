# True-talent xwOBA via empirical Bayes — "how good is this hitter, really?"

Date: 2026-07-18 · No re-fit. Reproduce: `.venv/bin/python scripts/run_talent.py`
→ `results/talent/{figures/, talent_table.parquet, talent_metrics.json}`.

**Goal.** The object for *analyzing a batter's true talent*: a per-batter-season xwOBA
that is (1) **regressed for sample size** — a hot 120-PA line is pulled toward the league
until the PAs earn it — and (2) carries an **interval that narrows with PA** and is
centered on where the hitter's talent actually sits. This is the piece the earlier work
was missing: v0's BART posterior interval is a *surface* band, ~flat in PA (Task A); the
PA-bootstrap band (`results/player_ci/`) narrows correctly but is centered on the **raw**
number, so it over-credits hot small samples (Trout 2024, 125 PA: raw .407, band
[.336,.480] — true talent is nowhere near .407).

**Construction.** Gaussian–Gaussian empirical Bayes (James–Stein / Efron–Morris), no BART:
1. Each PA gets an xwOBA value — batted ball → `estimated_woba_using_speedangle` (fallback
   `woba_value`), walk/K/HBP → deterministic `woba_value`. Raw xwOBA = Σvalue / Σdenom.
2. Sampling SE of that mean = `sd(per-PA values) / √n`; `se² ` is the within-player variance.
3. **Per season** (league offense drifts year to year), fit hyperparameters on the stable
   population (PA ≥ 100): between-player spread `τ²` by method of moments (observed variance
   of raw − mean SE²) and the precision-weighted league mean `μ`.
4. Shrink **every** player-season: `reliability = τ²/(τ²+se²)`,
   `θ̂ = μ + reliability·(raw − μ)`, posterior variance `= reliability·se²`, 90% interval
   `θ̂ ± 1.645·√(post_var)`. Reliability rises with PA (`figures/reliability_vs_pa.png`),
   so small samples shrink hard and their intervals are wide.

Per season (`talent_metrics.json`), 2,636 player-seasons total:

| season | n | μ | τ | median reliability |
|---|---|---|---|---|
| 2022 | 678 | 0.305 | 0.031 | 0.65 |
| 2023 | 647 | 0.314 | 0.031 | 0.64 |
| 2024 | 645 | 0.310 | 0.031 | 0.65 |
| 2025 | 666 | 0.318 | 0.032 | 0.65 |

**It behaves like a true-talent estimate.** Small samples are pulled toward the mean and
the interval narrows with PA (median 90% width 0.083 at 30–100 PA → 0.047 at 450+ PA):

| batter-season | PA | raw | → talent (90% interval) | reliability |
|---|---|---|---|---|
| Khalil Lee 2022 (extreme hot) | 2 | .911 | **.305** [.254,.357] | 0.001 |
| Mike Trout 2024 (hot) | 125 | .407 | **.341** [.299,.383] | 0.31 |
| Austin Hedges 2024 (cold) | 143 | .168 | **.212** [.184,.240] | 0.70 |

(A boom-or-bust hitter like Trout has higher per-PA variance → *lower* reliability than a
low-variance contact hitter at similar PA, so he is regressed harder — the estimate uses
each player's own noisiness, not just their PA count.)

## Validation — does shrinkage predict next season better than raw / Savant?

The decisive test (`validation` in `talent_metrics.json`): does season-T xwOBA predict
season-(T+1) **actual** wOBA? Pearson r vs next-season wOBA, players with a stable T+1
sample (≥100 PA):

| population | n | **EB talent** r | raw r | Savant r |
|---|---|---|---|---|
| pooled, PA_T ≥ 100 (vs 0.487 anchor) | 1060 | **0.489** | 0.484 | 0.491 |
| pooled, PA_T ≥ 30 (admits low-PA) | 1173 | **0.467** | 0.445 | 0.452 |

- **EB talent beats raw xwOBA** in both populations, and **beats Savant once genuinely
  low-PA seasons are included** (0.467 vs 0.452) — the regime shrinkage is built for.
- Against the frozen **r 0.487 Savant anchor** (100+ PA both years), EB talent (0.489) is
  at **parity** with Savant (0.491) — same statistical tie v0 established, now with a
  sample-size-honest *center* instead of a raw one.
- Calibrated RMSE: EB talent 0.0345 = Savant 0.0345 < raw 0.0346 (pooled PA_T ≥ 100).

**Why the win is a *pooled* effect, not a per-band one (important, and not a bug).** Inside
a narrow PA band reliability is ~constant, so `θ̂ = μ(1−c)+c·raw` is ~**affine** in raw and
Pearson r is affine-invariant — shrinkage *cannot* move within-band r much, and the tiny
per-band talent−raw gaps (`by_band` in the metrics) are noise (n≈50–60, correlation SE
≈0.13). Shrinkage's real payoff is **variance-compression across a heterogeneous-PA
population**: it tames the wild low-PA raws (which fan from .05 to .9) so the pooled cloud
tracks next-season wOBA better. That is exactly what the pooled numbers show. An earlier
"low-PA subset" test (PA<250 within the PA≥100-both-years filter) is doubly the wrong lens
— it is censored at 100 PA *and* affine-blind; we report pooled r plus the by-band table
instead.

## Limitations carried into Phase 2

1. **The prior is the flat league mean.** A good-contact / low-PA hitter is shrunk toward
   *league average*, not toward their **contact-implied** xwOBA. Phase 2 replaces μ with a
   BART contact-quality prediction so a rookie barreling the ball regresses toward a
   barrel-hitter's xwOBA — this is the job that finally gives BART a defensible role.
2. **The interval is the estimation (talent) interval only.** It does not yet fold in
   BART's *surface* term (how well we know each ball's true EV, ~flat ≈0.056). The fullest
   band is `width ≈ √(talent_var + surface_var)`: estimation dominates at low PA, the BART
   surface term at high PA (`results/player_ci/NOTES.md`). Phase 2 prerequisite: persist
   per-event model EVs (`event_ev.parquet`), which needs one Stage-C re-fit.
3. **Degenerate SE at tiny PA.** A player whose ≤5 PAs happened to have identical outcomes
   has sample SD ≈ 0 → SE ≈ 0 → reliability ≈ 1 (no shrinkage), visible as a few points at
   the top-left of `figures/reliability_vs_pa.png`. Harmless here (all validation filters
   PA ≥ 100 in the target year, and these carry no weight), but Phase 2 should floor `se²`
   or use a Beta-Binomial / count-based reliability so tiny samples cannot look certain.
