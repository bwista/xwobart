# Handoff — design Stage 4 (xwobart Phase 2): the surface⊕sampling interval + coverage validation

Paste the block below into a fresh conversation in `/Users/jweinga/Documents/python/xwobart`.

Stage 4 has **no plan or spec yet** — it was always scoped to get its own design cycle once
Stage 3's numbers existed. They exist now. So this kicks off *design*, not execution: brainstorm →
spec → plan → execute.

---

Use the `superpowers-extended-cc:brainstorming` skill to design **Phase 2 Stage 4** of xwobart.

## The goal in one paragraph

xwobart's actual deliverable — settled at the "product reframe" and not built by anything so far —
is a **per-player-season xwOBA talent interval that is honest about sample size**: wide at 40 PA,
tightening toward a floor as PA grows, with **validated coverage** (a claimed 90% interval contains
the truth ~90% of the time). Phase 1 gave sample-size-honest *centers*; Task A showed v0's BART
posterior interval is the wrong object (it is a *surface* band, ~flat in PA, not a sampling band);
`results/player_ci/` built the correct *sampling* band (proper 1/√PA) but centered on the raw number.
Stage 4 is the synthesis: combine the **sampling** term (dominant at low PA) with the **surface**
term (dominant at high PA) in quadrature — `width ≈ √(sampling² + surface²)` — and prove the result
is calibrated. This is the one thing Phase 1 structurally cannot produce.

## Read first (design of record + what just happened)

1. `results/RESULTS.md` — the whole "Phase 2 Stage 3 — spray surface" section, and the
   "Sampler reproducibility" section above it. **Stage 3's headline gate E1 FAILED**: the 5-feature
   spray surface did not beat the 3-feature v0 anchor (the gain landed inside the run-to-run noise
   floor). That failure is *not* a blocker for Stage 4 — see "What Stage 3 leaves you" below.
2. `results/player_ci/NOTES.md` — the combined-width design (`√(sampling² + surface²)`), why the
   sampling band is model-agnostic, and the ~1.8×-too-tight gap at 100 PA this is meant to fix.
3. `results/talent/NOTES.md` and `results/talent2/NOTES.md` — the EB and joint-MVN talent models
   Stage 4 extends. Note the affine-invariance lesson (per-band r cannot see shrinkage; report pooled
   RMSE + by-band).
4. `docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md` — §"Risks" 3 in particular
   (the correlated-surface-error caveat below is from there).

## What Stage 3 leaves you — the prerequisite is finally on disk

The spray refit failed its ELPD goal but **succeeded at producing the artifact Stage 4 needs**: it
persisted the model's per-event value draws, which are its *surface uncertainty*, and they are
mechanically sound (the OOS gate E4 passed at corr 0.997). In `results/stage_C_spray/`:

- `ev_draws_{train,holdout}.npy` — **(200, 363595)** and **(200, 122006)** float32, per-event value
  draws, positionally row-aligned with…
- `ev_draws_keys_{train,holdout}.parquet` — `row`, `batter`, `season`, `woba_denom`, the 5 features,
  `hc_imputed`. Alignment is positional (axis 1 of the `.npy` ↔ `row` order of the parquet);
  `metrics.json` stamps a `batter_order_digest` per split.
- `lppd_i_holdout.npy`, `all_trees.pkl` (27.6 MB, reloadable — `scripts/marginalize_spray.py` shows
  the predict-from-pickle path).

## The exact code seam (verified 2026-07-20)

- **The surface term goes into `S[0,0]`.** `src/talent2.py:bootstrap_S` (lines ~99–107) builds the
  per-player measurement covariance `S_i` by resampling PAs; `S[0,0]` is currently the **sampling**
  variance of the player-season xwOBA (floored at `FLOOR_SD_PER_PA²/n`). Stage 4 *adds* a surface
  term: `S[0,0] += surface_var_i`. The joint-MVN posterior at `talent2.py:183` (`V = Σ − Σ(Σ+S_i)⁻¹Σ`)
  then propagates it into the talent interval automatically — you do not touch the posterior math,
  only what feeds `S_i`.
- **`surface_var_i` already has machinery.** `src/rollup.py:player_rollup` rolls the (200, n) draws up
  to per-draw player-season xwOBA and returns `xwoba_sd`; `xwoba_sd²` per player-season *is* the
  surface term. So the core is: roll up the persisted draws → join `xwoba_sd²` onto the measurements →
  add to `S[0,0]`.
- **Level 2's center currently comes from public Savant.** `src/talent2.py:build_pa_measurements:42`
  reads `estimated_woba_using_speedangle`. Whether Stage 4 also switches the *center* to the model's
  own rollup is design decision D4 below.

## Open design decisions the brainstorm must resolve (this is the substance)

- **D1 — which surface's draws.** The draws on disk are from the *spray* surface, which is marginally
  worse than v0 on ELPD (within noise). v0 did **not** persist draws (persistence is gated on
  `variant == "spray"` in `run_v0.py`). Options: (a) use the spray draws as-is; (b) add v0 to the
  persist gate and do one ~30-min v0 run for a best-surface version; (c) regenerate from
  `all_trees.pkl`. This also interacts with `scripts/capacity_experiment.py` (built, not yet run) — if
  that runs and settles the surface, use its winner. For a coverage *methodology*, the exact surface
  barely matters; for the shipped product it might.
- **D2 — aggregating draws to `surface_var_i`.** `player_rollup`'s `xwoba_sd²` is the obvious estimator,
  but confirm it is the *within-player* spread you want, and mind the caveat below.
- **D3 — coverage against WHAT truth (the single most important decision).** The interval claims to
  contain a player's *true talent*, which is unobservable. Testable proxies each have a confound:
  (a) does the season-T interval contain season-(T+1) *actual* wOBA? — clean target, but conflates
  interval coverage with genuine year-to-year talent drift, so a well-calibrated talent interval will
  *under*-cover next-season wOBA unless drift is modeled; (b) split each player's PAs, build the
  interval on one half, check coverage of the other half's xwOBA — a clean test of the
  sampling⊕surface statement with no drift confound, but needs enough PA to split. The plan said
  "50/80/90% coverage by PA band within ±5pp" but never specified the target. **Resolve this first —
  it shapes everything else.**
- **D4 — does the interval's center change?** If the surface *variance* comes from the model's draws
  but the *center* stays Savant's xwOBA, there is a coherence mismatch (uncertainty of a quantity you
  are not using). Switching the center to the v0 model rollup means teaching `build_pa_measurements`
  to take model values instead of `estimated_woba_using_speedangle`. But the Stage-3 rollup A/B
  (`results/rollup_ab/`) found the model rollups do *not* out-predict Savant, so this is a real
  tension, not a free win.

## Load-bearing caveats (do not design around these silently)

- **Correlated surface errors (design Risk 3).** The between-draw variance is exactly right for
  *per-player* intervals, but surface errors are **correlated across players in the same feature
  region** — so these draws must **not** be used for any league-aggregate coverage claim without a
  per-draw refit variant. Keep the coverage test per-player.
- **The draws are one fit's *within-fit* posterior spread.** pymc-bart 0.12 is not reproducible across
  processes (RESULTS.md), so 200 draws from one fit capture posterior uncertainty of *that* surface,
  not across-fit variability. That is the right term for a per-player surface interval; just name it.
- **Affine-invariance.** Report pooled RMSE + a by-band table, never per-band r alone (see the talent
  NOTES).

## Scope

**IN:** fold `surface_var_i` into `S[0,0]`; construct the combined sampling⊕surface talent interval;
validate 50/80/90% coverage by PA band within ±5pp. **OUT (explicit follow-ons, their own cycles):**
multi-season + age pooling at Level 2 (design review flagged this as possibly a *bigger* lever than
spray was); the capacity experiment (`scripts/capacity_experiment.py`, already built) is an
independent surface side-quest and not a Stage 4 dependency.

Start by exploring the four docs above, then work the open decisions one at a time. D3 first.

---

## Out of scope for this handoff

Running `scripts/capacity_experiment.py` (the equal-capacity spray-vs-v0 retest) — it is built and
smoke-tested but not run, and its outcome only changes *which* surface's draws Stage 4 uses (D1), not
the Stage 4 method. Run it whenever; it does not block the design.
