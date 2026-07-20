# Handoff — write the Stage 2+3 plan (xwobart Phase 2)

Paste the block below into a fresh conversation in `/Users/jweinga/Documents/python/xwobart`.

---

Use the `superpowers-extended-cc:write-plan` skill to write an executable implementation plan for
**Phase 2 Stages 2+3** of this project. Do not implement anything — the deliverable is the plan
document plus its `.tasks.json` tracker. The design is already settled and reviewer-approved, so
this needs planning, not a fresh brainstorm.

## Read these first, in this order

1. `docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md` — **the design of record.**
   Stages 2–4 are specified here; §"Recommended spec" and housekeeping item 6 carry the substance.
2. `docs/superpowers/plans/2026-07-19-xwobart-phase2-level2-talent.md` — the Stage 1 plan. **Use it
   as the template and the quality bar**: up-front success gates with HARD/DIAGNOSTIC/PROTOCOL
   labels, complete code for the core math inline, TDD tasks with explicit failing-test steps,
   frozen numeric anchors as regression gates, a stop-and-report rule if the thesis gate fails.
3. `results/talent2/NOTES.md` and `results/RESULTS.md` — what Stage 1 actually produced.
4. `results/RESULTS.md` §"Deviations from spec/plan" — **the pymc-bart 0.12 traps are recorded here
   and will bite Stage 3 if ignored.** Especially: `pm.set_data` + `sample_posterior_predictive`
   silently FREEZES `mu` (returns the in-sample trace regardless of `X_new` — a wrong answer, not
   an exception); use the stored-trees predictor `_sample_posterior`, which returns `(S, n, K)` and
   must be transposed to the project's `(S, K, n)` convention. Scripts that fit must guard
   executable code under `if __name__ == "__main__"` (spawn re-import). BART `mu` R-hat is
   structurally high and is NOT a convergence signal — gate on `verify_oos_mechanism` (corr ~1.0).
5. `docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md` — read the supersession block at
   the top only. Its two candidate designs were killed; the body is history.

## Where things stand

Phase 2 Stage 1 shipped today (`origin/main` @ `7c1e9cb`): `src/talent2.py` + `scripts/run_talent2.py`
+ `results/talent2/`, a joint-MVN talent model over (xwOBA, avg EV, barrel rate). It passes its
hard gates — PA≥30 r **0.4698** vs Phase-1 0.4669, PA≥100 **0.4908** (ties public Savant) — with the
gain concentrated at low PA (+0.0718 at 30–60 PA, −0.0000 at 250+) and the shared-noise tripwire
clean at −0.0130. It is honestly documented as *not statistically established*: the paired-bootstrap
CI straddles zero and the held-out confirm season reverses sign.

Stage 1 needed no cache rebuild and no BART refit. **Stages 2+3 need both**, which is why they
belong in one plan: the rebuild's only consumer is the refit, and the sign-QC gate and the ELPD gate
want to live in the same document.

## Scope to plan

**Stage 2 — cache rebuild.** Add `hc_x`, `hc_y`, `stand` to `KEEP_COLUMNS` in `src/data.py:15` and
rebuild the slim caches (`build_season_caches(..., force=True)`; `scripts/run_v0.py` already exposes
`--force-data`). Derive pull-relative spray angle:
`φ_raw = atan2(hc_x − 125.42, 198.27 − hc_y)` in degrees, **mirrored by `stand`** so it is
pull-relative — and keep `stand` as a *separate* feature so BART can still recover raw direction
(park asymmetries act on raw direction, batter skill acts pull-relative). Housekeeping the design
calls out explicitly: `stand` is per-event (switch hitters); quantify `hc_x`/`hc_y` missingness and,
if small, impute the league conditional mean given EV×LA plus a missing flag.

**The sign QC is a hard gate, not a smoke test** — known pull hitters must cluster on the same side
for *both* handednesses. Getting this backwards silently mirrors half the league and would poison
Stage 3 invisibly. Design a gate that would actually catch it.

**Stage 3 — one 5-feature BART surface refit.** Features become
`launch_speed, launch_angle, spray_pull, stand, sprint_speed` (`FEATURES` in `src/prep.py:10`).
Same machinery, same train-subsample protocol and holdout as v0 **so the anchor stays comparable**
(Stage C config: subsample 100k, m_trees 50, tune 500, draws 500, chains 2). No batter terms, no
player aggregates.

- **The gate: ELPD on the 2025 holdout must decisively beat −80107 ± 244** (122k events). The design
  expects an unambiguous beat — spray is enormously informative for hit type given EV/LA — and says
  to treat a non-beat as a red flag, so plan for that branch.
- **Persist ~100–200 thinned per-event value draws** `v^(s)(x_e)` as float32 (≈290 MB). This is the
  Stage 4 prerequisite; Stage 4 folds that variance into `S_i[0,0]` and finally makes interval
  coverage testable. Do not let this get dropped — it is the whole reason the refit is worth doing
  once rather than twice.
- Diagnostics: PDP/importance for spray (HR band in LA×spray), confirm sprint speed's contribution
  migrates toward pulled grounders, and watch the rare 3B class (spray should *help* it).
- Compute both rollups — spray-conditioned and spray-marginalized (replace `v(x_e)` with its
  league-average over spray given EV×LA×stand; no refit needed). The design flags a real
  descriptive/predictive inversion risk here: conditioning on per-ball direction credits spray
  *luck*. Plan the A/B, let next-season RMSE pick, and do not assume the conditioned rollup wins.

Stage 4 is **out of scope** — it gets its own plan once Stage 3's numbers exist.

## Repo conventions

- Work happens directly on `main`, no worktree. Commit per task; push only when asked.
- Run everything from the repo root with `.venv/bin/...`. Tests are plain pytest (`48 passing`
  currently; a new task should state the expected new total).
- Pure logic in `src/`, orchestration + figures in `scripts/`, results + `NOTES.md` per directory
  under `results/`. Polars throughout (pandas only at boundaries).
- Statcast source: env `STATCAST_PATH`, monthly parquet cache; the project depends on
  `-e ../kinferencetoolkit` for `pipeline.statcast_loader`.
- **Runtime warning:** Stage 3 is the expensive step (~27 min for the v0 Stage-C fit, vs Stage 1's
  7 seconds). The plan should say so up front and put the cheap sign-QC gate *before* the refit, so
  a mirrored-spray bug cannot cost half an hour.

## Two things to decide while planning, and flag to me

1. Whether Stage 2's rebuild should be gated behind a check that the *existing* v0/Phase-1/Stage-1
   artifacts still reproduce afterward — adding columns shouldn't change existing rows, but that is
   an assumption worth a cheap regression gate given how many frozen anchors now depend on the
   caches (2,636 player-seasons; r 0.4886/0.4669; ELPD −80107).
2. Whether Stage 3 should re-run `scripts/run_talent2.py` at the end to confirm Level 2 is unchanged
   by the surface swap, or whether that belongs in Stage 4.
