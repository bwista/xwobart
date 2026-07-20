# Handoff — execute the Stage 2+3 plan (xwobart Phase 2)

Paste the block below into a fresh conversation in `/Users/jweinga/Documents/python/xwobart`.

---

Use the `superpowers-extended-cc:executing-plans` skill to execute
`docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md`.

The plan is complete, reviewed twice (two blockers found and fixed), and its `.tasks.json` tracker
sits beside it. Ten tasks, 45 steps. Work directly on `main`, no worktree; commit per task; push only
when asked. Everything runs from the repo root with `.venv/bin/...`.

## Read first

1. The plan itself — front matter through the "Success gates" tables before touching anything. The
   gate tables (R1–R6, S1–S6, E1–E8) are the contract.
2. `results/RESULTS.md` §"Deviations from spec/plan" — the pymc-bart 0.12 traps. Especially:
   `pm.set_data` + `sample_posterior_predictive` silently FREEZES `mu` (returns the in-sample trace
   regardless of `X_new` — a wrong answer, not an exception); BART `mu` R-hat is structurally high
   and is NOT a convergence signal.

## Three things that will bite if you skip them

- **Task 4 is destructive.** It rebuilds the slim caches in place. It backs them up to
  `data/raw/prerebuild/` first and gates on byte-level reproduction of the pre-existing 15 columns.
  If R1/R2/R5 fail, the *upstream* Statcast cache moved — that is not a code bug, and every frozen
  anchor in the repo (2,636 player-seasons; r 0.4886/0.4669; ELPD −80107 over 122,006 events) is
  invalid until reconciled. **Stop and report; do not re-freeze the anchors.**
- **Task 8 costs ~60 minutes and must run exactly once.** Fit and spray-marginalization share one
  invocation because the fitted trees live on the in-memory model object and cannot be recovered
  from `idata.nc`. Tasks 1–7 must be fully green first — 61 tests passing, R1–R6 PASS, S1–S6 PASS,
  and the ~3-minute Stage-A spray smoke (Step 7.4) clean. That smoke exercises every line of the
  60-minute path; it is not optional.
- **E1 is the thesis gate**: holdout ELPD must beat −80107.495 by ≥1000 nats over exactly 122,006
  events. If it fails, work the plan's 5-item checklist and **stop and report**. Do not raise
  `m_trees`, `draws`, or the subsample — that breaks protocol parity, which is the entire reason the
  v0 orchestrator is being reused rather than forked.

## Decisions already made (do not relitigate)

1. The rebuild is gated on byte-level reproduction, not just row counts.
2. `run_talent2.py` re-runs in Stage 2, not Stage 3 — it consumes public Savant per-event xwOBA and
   never touches the surface, so the refit provably cannot move it.
3. hc missingness (0.034–0.043% of BBE) is imputed and flagged, but the flag is an **audit column,
   not a BART feature** — rows are never dropped, because that would move the holdout off 122,006
   and break the ELPD anchor.

## Verified during planning (trust these numbers)

- Spray sign convention confirmed on all four seasons: `φ_raw = atan2(hc_x − 125.42, 198.27 − hc_y)`
  is negative for left field; mirroring by `stand` gives mean pull positive for **both** hands
  (L +6.8…+7.5, R +3.2…+3.6), HRs at +16…+20 with ~80% pull-side.
- BBE per season after `filter_bbe`: 118,891 / 122,070 / 122,634 / 122,006 — which reproduce v0's
  `predict_rows` (363,595 train, 122,006 holdout) exactly.
- Gate S5 was validated by *simulating* the bug it exists to catch (resolving `stand` per
  player-season instead of per event): S1, S2 and S3 all still pass; only S5 fails, at −3.969 on the
  RHB subgroup of the 65 switch-hitter batter-seasons. Write S5 exactly as specified — the obvious
  "count switch hitters" version is vacuous.

## Out of scope

Stage 4 — folding the persisted draw variance into `S_i[0,0]`, pushing the A/B-winning rollup
through Level 2, and interval-coverage validation. It gets its own plan once Stage 3's numbers exist.
