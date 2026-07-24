# Handoff — condense the review notebooks (7 → 3)

Paste the block below into a fresh conversation at the repo root
(`/Users/jweinga/Documents/python/xwobart`).

---

Restructure the xwobart review notebooks from **7 chronological parts into 3 thematic ones**, and in
the same pass absorb two findings that landed *after* the notebooks were written. This is a
**presentation** task — no model re-fits. Notebooks read already-committed artifacts under `results/`
and run in seconds.

## Why (the problem to fix)

The series is shaped by **chronology** — one part per stage of work — so it grew with effort rather
than with findings. There are ~5 real findings spread across 7 notebooks, and `results/RESULTS.md`
plus `results/*/NOTES.md` already carry the blow-by-blow ledger, so the notebooks duplicate it.

**Part 6 (spray) is the clearest symptom: spray isn't its own idea.** Its *description* half is a
**surface** finding and its *forecast* half is a **forecast** finding. Dissolving it into the two arcs
it actually belongs to absorbs the new results **without adding a part** — and kills the chronological
framing that caused the bloat.

## The target structure

| New | Absorbs | The finding it must deliver |
|---|---|---|
| **01 — The surface and its ceiling** | old 01 + 02 + 06 + the **new** m=200 capacity result | v0 replicates Savant (player r **0.956**) but sits at its **information ceiling for prediction** (next-season parity, r **0.481 vs 0.487**). Spray — the information Savant lacks — genuinely improves *description* at adequate capacity (m=200: **+3,017 nats** paired vs a **36.7-nat** noise floor, reversing the m=50 negative as **capacity dilution**), yet does **not** breach the prediction wall (spray rollups still lose to v0). Calibration regresses under spray (ECE **0.0369** vs **0.0277**). |
| **02 — Uncertainty and true talent** | old 03 + 04 + 05 | v0's posterior band is the **wrong object** (flat in PA); a PA-bootstrap narrows correctly and the two cross near **400 PA**. EB shrinkage turns that into a true-talent estimate that beats raw everywhere and edges Savant once low-PA seasons are admitted (**0.467 vs 0.452**). Level 2 puts the fast-stabilizing peripherals in the prior: **+0.072 r at 30–60 PA**, ~nothing at 250+, pooled effect small and one holdout season disagrees — with the shared-noise tripwire coming back clean. |
| **03 — The product: forecasting the rest of a season** | old 07 + the **new** rung-b spray negative | From a hitter's first *k* PAs, forecast his final-season xwOBA with a range. Pooled RMSE **0.0220** beats naive / Marcel / single-season L2 (bootstrap CIs exclude 0); G5 reduces to Phase 1 exactly (**5.6e-17**); **G4 calibration fails** — 50/80% intervals run narrow, worst at short runway. And **spray adds nothing here**: pull tendency is forecast-redundant (**ΔR² ≤ +0.0022** beyond early xwOBA/EV/barrel). |

Suggested filenames: `01_surface_and_ceiling.ipynb`, `02_uncertainty_and_talent.ipynb`,
`03_forecast.ipynb`. **Delete the old `01`–`07` in the same commit** — git history preserves them.

## The two NEW findings to absorb

1. **m=200 capacity experiment → new notebook 01.** Source of record: `results/RESULTS.md` §"Spray at
   matched capacity — the m=200 result". **Guard against the artifact**
   `results/capacity_C_m200/capacity_metrics.json`, not the prose. Figures available:
   `results/stage_C_spray_m200/figures/pdp_la_spray_hr.png` (spray *is* learned),
   `.../calibration_reliability.png`, `.../variable_importance.png`, plus the v0 anchor/replicate
   equivalents under `results/stage_C_m200{a,b}/figures/`. Key framing: the m=50 "negative" was
   **capacity dilution**, now confirmed — and the noise floor collapsed **267 → 37 nats** because
   *both* models were under-capacity at m=50 (v0 gained +4,990 nats, spray +7,758, going 50 → 200).
2. **Rung-b spray pre-check → new notebook 03.** Source: `results/talent3/NOTES.md` §"Rung b — spray
   peripheral: a documented negative". **Guard against** `results/talent3/precheck_pull.json`.
   Framing: **description-yes / forecast-no** — pull tendency is redundant with EV/barrel, which
   capture the same fast-stabilizing power skill. It is a **well-powered** null (at n=1,945, k=50, an
   injected ΔR²≈0.04 is detected easily) and season-FE robust.

## Conventions contract (established in commit `a9454da` — do not drift)

- **Setup cell**: copy the current notebook 01's first cell verbatim — it locates the repo via
  `config.yaml` and imports `ROOT, RESULTS, jload, show_fig` from `notebooks/nb_helpers.py` (polars
  dtype-hiding lives there; do not re-add per-notebook config).
- **Figures**: `show_fig(rel, caption="...")` — auto-sizes to native width capped at 980 px, italic
  caption underneath. Every figure gets a one-line *what to look for* in the markdown **above** it;
  interpretation after only when it adds something.
- **Numbers**: every number quoted in prose must be checked against a **committed artifact**. Each
  notebook **ends with a guard cell** (`# guard: ...`) asserting the headline numbers and printing
  `prose numbers still match the artifacts`. Copy the style from the existing notebooks.
- **Voice**: first person, plain language; each part opens by answering the question the previous part
  raised and closes by raising the next one.
- **Process**: author each notebook as complete nbformat JSON via a Python script (they're new files).
  Execute headless (`nbclient`, kernel `python3`, cwd `notebooks/`), write back **with outputs
  embedded**, then scan for zero `error` outputs before committing.
- **No re-fits.** Notebooks only read committed artifacts under `results/`; they must run in seconds
  with no model traces present.

## Also update, same commit

- `notebooks/README.md` — the parts table (3 rows), the "how to run" text, and the arc paragraphs.
- Root `README.md` §"The notebooks" and §"The story so far" — collapse the 7-arc list to match the
  3-part structure and the resolved spray verdict.

## Definition of done

- Three notebooks execute headless from a fresh-clone artifact state, **zero error outputs**, guard
  cells print their pass line.
- Old `01`–`07` deleted; both READMEs updated; everything committed **with outputs embedded** and pushed.
- No number in prose that isn't guarded against a committed artifact.

## Watch out

- **Root README arc 6 is stale** — it still says the capacity follow-up is "written but not yet run".
  It ran, and spray **won** for description. Fix it.
- The old notebook 06 quotes a **267-nat** noise floor and a Stage-A triple that live **only in
  RESULTS.md prose** (no committed `metrics.json`). At m=200 the equivalents **are** committed in
  `capacity_C_m200/capacity_metrics.json` — prefer the guardable ones, and hardcode any m=50
  reference with a comment pointing at RESULTS.md.
- `results/stage_C_spray/` (m=50) still has the older 1×5-strip calibration render; the m=200 one is fine.
- Old notebook 07 is **1.1 MB** with outputs embedded — merging carelessly will bloat the new files.
  Keep only the figures that earn their place.
- Do **not** re-litigate spray: it is resolved (description-yes / forecast-no). See
  `docs/superpowers/handoffs/2026-07-23-forecast-open-levers-handoff.md` for the open levers.
