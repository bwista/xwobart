# xwobart Phase 2 — BART-informed contact-quality prior (ROADMAP)

> ## ⚠️ SUPERSEDED 2026-07-19 — read this block before using anything below
>
> This roadmap's two candidate designs were both killed, one empirically and one at design
> review. The text below is kept **for history only**; it is not the plan of record.
>
> - **"Shrink toward the model's own xwOBA" — killed empirically.** A structural no-op: the
>   BART surface is a function of the same batted balls the raw number is built from, so the
>   residual between-player variance it leaves is τ_resid² ≈ 1e-4. There is nothing to shrink
>   toward that the raw number does not already contain.
> - **Event-level batter intercepts — rejected at design review.** That channel has
>   year-to-year r ≈ 0.12. Modeling it is modeling noise.
> - **What replaced them.** At 30–100 PA, peripherals out-predict raw xwOBA next season
>   (barrel r 0.244, EV r 0.227 vs raw xwOBA r 0.179), so they belong in the **prior**,
>   entering at the **player-season** level rather than the event level.
>
> **Plan of record:** `docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md`
> (two-level modular design; joint MVN at Level 2, no event-level batter effects).
>
> **Stage 1 — SHIPPED 2026-07-19.** Joint-MVN talent model over (xwOBA, avg EV, barrel rate):
> `docs/superpowers/plans/2026-07-19-xwobart-phase2-level2-talent.md` → `src/talent2.py`,
> `scripts/run_talent2.py`, `results/talent2/`. All hard gates pass (PA ≥ 30 r 0.4698 vs
> Phase-1 0.4669; PA ≥ 100 0.4908, tying Savant; shared-noise tripwire clean at −0.0130), with
> the gain unestablished — bootstrap CI straddles zero and the confirm season reverses sign.
> Full accounting in `results/talent2/NOTES.md`.
>
> **Stages 2–4 — not yet planned.** (2) cache rebuild adding `hc_x`/`hc_y`/`stand` with
> pull-mirrored spray angle and sign QC; (3) one 5-feature BART surface refit, ELPD against the
> −80107 anchor; (4) fold the surface-draw variance into `S_i[0,0]`, spray-conditioned vs
> spray-marginalized rollup A/B, and 50/80/90% interval-coverage validation by PA band.
>
> **The per-event persistence prerequisite below MOVES to Stage 3** and is upgraded: it must
> persist per-event value **draws** (~100–200 thinned), not just means, so Stage 4 can fold the
> surface variance in.
>
> Multi-season + age pooling at Level 2 is the flagged follow-on after Stage 4 — and per
> Stage 1's notes, likely worth more than the peripherals were.

> **Status: roadmap, NOT an executable plan.** This is the design and the decisions to
> settle, not bite-sized tasks. Phase 2 requires a modeling decision and one BART re-fit, so
> it must be run as its own **brainstorm → spec → plan → execute** cycle (like v0). Extracted
> 2026-07-18 from the Phase-1 plan (`2026-07-18-xwobart-talent-estimates.md`) once Phase 1
> shipped, so the completed plan stays focused on what was built.

## Where Phase 1 left off (the baseline to beat)

Phase 1 (`src/talent.py`, `scripts/run_talent.py`, `results/talent/`) built a per-batter-season
**empirical-Bayes true-talent xwOBA**: shrink each player's raw xwOBA toward the **season league
mean** by its reliability `τ²/(τ²+se²)`, with a calibrated interval that narrows with PA. It works
and is at parity with Savant:

| population (predict next-season actual wOBA, r) | n | EB talent | raw | Savant |
|---|---|---|---|---|
| pooled, PA_T ≥ 100 (vs the 0.487 anchor) | 1060 | **0.489** | 0.484 | 0.491 |
| pooled, PA_T ≥ 30 (admits low-PA) | 1173 | **0.467** | 0.445 | 0.452 |

Phase 1 leaves **two limitations**, and Phase 2 exists to close the first (and can close the second):

1. **The prior is the flat league mean.** A good-contact / low-PA hitter is shrunk toward *league
   average*, not toward what their **contact quality** implies. A rookie with 80 PA of barrels
   should regress toward a barrel-hitter's xwOBA, not toward .310.
2. **The interval is estimation-only.** It does not fold in BART's *surface* term (how well we know
   each ball's true expected value, ~flat ≈ 0.056). The fullest band combines both.

## Why Phase 2

Replace the flat prior mean with a **BART contact-quality prediction**, so a player's own PAs
update a contact-informed prior instead of the league average. This is the job that finally gives
the BART model a clear, defensible role — Phase 1 is deliberately model-agnostic, and v0's posterior
interval was the wrong object (surface uncertainty, not sample size). The payoff is directly
testable: a contact-informed prior should beat Phase-1 league-mean EB **specifically for players
whose contact quality diverges from their small-sample results** (barreling the ball with unlucky
outcomes over ~100 PA, or the reverse).

## Prerequisite (blocks everything in Phase 2)

**Persist per-event model expected values during a fit.** The fitted BART trees are not saved in
`idata.nc` (they live only in `model["mu"].owner.op.all_trees` in memory), so the model's per-BBE
EVs cannot be recovered without a re-fit. Add an option to `scripts/run_v0.py` / `src/model.py` to
write per-event `ev_mean` (holdout **and** train) to `results/stage_C/event_ev.parquet`, keyed by
`(batter, season, event index)`. One re-fit of Stage C (~27 min) covers it. See v0 status memory /
`results/RESULTS.md` "Deviations" for the stored-trees predictor gotchas (`pm.set_data` +
`sample_posterior_predictive` silently freezes `mu`; use `_sample_posterior`).

## Design decision to settle in the Phase-2 brainstorm — two variants

### 1. Two-stage (recommended first; cheaper, reuses v0 and all of `src.talent`)

- **Stage 1 — contact-implied prior mean `m_i`:** the player-season's mean of the model's per-event
  EVs over their BBE, plus the deterministic non-BBE values (walks/Ks/HBP), from the persisted
  `event_ev.parquet`.
- **Stage 2 — shrink toward `m_i` instead of the league mean:**
  `θ̂_i = m_i + reliability_i · (raw_i − m_i)`, with the between-player residual variance `τ_resid²`
  estimated by the *same* EB machinery run on the residuals `(raw_i − m_i)`. Reuses `eb_fit` /
  `eb_shrink` directly — only the center changes.
- **Validation:** must beat Phase-1 league-mean EB on next-season wOBA for the **contact-diverges /
  low-PA** subset (or produce a clear reason why not).

### 2. Full event-level hierarchical BART (heaviest; most principled)

- Add a **batter random intercept** to the latent categorical model in `src/model.py`:
  `mu = BART(contact features) + b_batter`, `b_batter ~ N(0, σ_b²)`. The model gains player identity
  and does partial pooling internally; per-player posteriors come straight out of the fit.
- **Cost:** a genuinely new, larger fit (thousands of batter effects); needs its own runtime/memory
  gate like v0's Stage C decision (full-train already risks OOM at ~14.5 GB `mu` on an 18 GB box).
  Only pursue if the two-stage variant proves the concept and the extra coherence is judged worth the
  fit cost.

## Combined interval (either variant)

Report the fullest interval as `width ≈ √(talent_var + surface_var)` — the Phase-1/Phase-2 estimation
(talent) variance **plus** BART's surface posterior variance (the flat ≈ 0.056 term from v0).
Estimation dominates at low PA; the BART surface term dominates at high PA. This is where v0's
posterior finally contributes legitimately instead of masquerading as the whole band (cf.
`results/player_ci/NOTES.md`, which already sketches the `√(sampling² + surface²)` construction).

Also carry over the Phase-1 edge case: **floor `se²`** (or move to a Beta-Binomial / count-based
reliability) so tiny samples of near-identical outcomes cannot report reliability ≈ 1.

## Phase-2 definition of done (mirrors v0 §15)

- The two-stage contact-prior estimate beats Phase-1 league-mean EB on next-season wOBA for the
  low-PA / contact-diverges subset (or a clear, documented reason why not).
- `event_ev.parquet` persisted and documented.
- The combined-interval construction implemented and its **calibration checked** — does the interval
  contain next-season wOBA at the nominal 90%?
- `results/RESULTS.md` updated with the model-comparison table.
- All unit tests green.

## How to start Phase 2

Run its own cycle: `/superpowers-extended-cc:brainstorm` on the two variants above → write a spec →
`/superpowers-extended-cc:write-plan` → `/superpowers-extended-cc:executing-plans`. Do the
prerequisite (persist per-event EVs) first, as a standalone re-fit, since it gates everything else.
