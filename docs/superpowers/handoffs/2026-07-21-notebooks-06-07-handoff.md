# Handoff — write review notebooks 06 (spray) + 07 (rest-of-season forecast)

Paste the block below into a fresh conversation at the repo root.

---

Write two new review notebooks continuing the `notebooks/01–05` series, then update both
READMEs and commit with outputs embedded. The analysis is already done and written up —
this is a *presentation* task: turn the spray-surface arc and the rest-of-season-forecast
arc into notebooks with the same voice, conventions, and guard-cell discipline as 01–05.

- `notebooks/06_spray_surface.ipynb` — "Part 6"
- `notebooks/07_ros_forecast.ipynb` — "Part 7"

## Read these first, in this order

1. `notebooks/01_v0_model_quality.ipynb` and `notebooks/05_level2_talent.ipynb` — the
   voice and the conventions (see the contract below). 05 is the quality bar for a
   notebook that argues against itself honestly.
2. `notebooks/nb_helpers.py` — the shared setup every notebook imports.
3. `results/RESULTS.md` — sections "Sampler reproducibility", "Phase 2 Stage 2",
   "Phase 2 Stage 3" (+ its Rollup A/B and persisted-draws subsections), and
   "Rest-of-season xwOBA forecast (rung a)". These are the source of record for 06/07.
4. `results/stage2_rebuild/NOTES.md`, `results/rollup_ab/NOTES.md`,
   `results/talent3/NOTES.md` — the per-layer deep dives.
5. Root `README.md` §"The story so far" arcs 6–7 — the short versions the notebooks
   must stay consistent with.

## Conventions contract (established in the 01–05 overhaul, commit `a9454da`)

- **Setup cell**: copy notebook 01's first cell verbatim — it locates the repo via
  `config.yaml` and imports `ROOT, RESULTS, jload, show_fig` from `nb_helpers`
  (polars dtype-hiding lives there; do not re-add per-notebook config).
- **Figures**: `show_fig(rel, caption="...")` — auto-sizes to native width capped at
  980 px, italic caption underneath. Every figure gets a one-line *what to look for* in
  the markdown ABOVE it; interpretation after only when it adds something.
- **Numbers**: every number quoted in prose must be checked against a committed artifact.
  Each notebook ENDS with a guard cell (`# guard: ...`) asserting the headline numbers
  and printing `prose numbers still match the artifacts` — copy the style from 01–05.
- **Voice**: first person, plain language, each part opens by answering the question the
  previous part raised and closes by raising the next one.
- **Process**: author each notebook as a complete nbformat JSON via a Python script (new
  files — no need for NotebookEdit; if you do use NotebookEdit on id-less cells, do all
  replaces before any insert: `cell-N` ids are positional and an insert shifts them).
  Execute headless (`nbclient`, kernel `python3`, cwd `notebooks/`), write back WITH
  outputs, then scan for zero `error` outputs before committing.

## Notebook 06 — Part 6: the spray surface (a documented negative)

This is the payoff of Part 2's promise ("the only way past parity is inputs Savant
doesn't have — spray and handedness") and item 3 of Part 5's next-steps list. The story
is a negative result told properly, and its centerpiece is *measuring the ruler first*.

Suggested beats:

1. **Rebuild + sign QC** (`results/stage2_rebuild/`): caches gain `hc_x/hc_y/stand`,
   `spray_pull` is mirrored per event so positive = pulled for both hands. R1–R6
   reproduction gates all PASS (R2's content digest byte-identical → every frozen anchor
   survives); S1–S6 sign gates all PASS (league pull L +6.84…+7.50 / R +3.23…+3.62,
   Schwarber +13.90, Mountcastle −5.16; S5 catches a modal-hand mirror via the 65
   switch-hitter seasons: +8.68 L / +5.60 R). Figures: `spray_by_hand.png`,
   `spray_hr_raw_direction.png`. Guard against `rebuild_report.json` + `spray_qc.json`.
2. **The noise floor** (RESULTS.md §"Sampler reproducibility"): pymc-bart 0.12 fits are
   not reproducible across processes even at fixed seed — three identical Stage-A runs
   spread 556 nats; a full Stage-C replicate landed **+267.1 nats** above the frozen
   anchor (−79,840.4 vs −80,107.5). Frame: before reading any spray number, measure how
   much two *identical* runs differ. NOTE: the replicate and Stage-A-triple numbers live
   only in RESULTS.md prose (no committed metrics.json for the replicate) — hardcode
   them with a comment pointing at RESULTS.md; the guard cell covers the committed side.
3. **The refit fails** (`results/stage_C_spray/metrics.json`): E1 FAIL — spray ELPD
   −79,876.3, i.e. +231.2 vs the anchor but *inside* the 267-nat noise floor, and −35.9
   vs the same-session replicate. E2 FAIL — ECE 0.0531 vs v0's 0.0422. Replication drops
   (event r 0.873 vs 0.911; player 0.908 vs 0.956). Crucially: the model *learned* spray
   — variable importance ranks it #3 of 5 (`indices [1, 0, 2, 3, 4]`), and
   `figures/pdp_la_spray_hr.png` shows P(HR) rising 0.20 → 0.37 from oppo to pull at
   EV 103 — it just doesn't buy held-out likelihood. Show the pdp figure; the
   3-row ELPD table (anchor / replicate / spray) is the notebook's key table.
   Caveats to state: `stage_C_spray/figures/calibration_reliability.png` is still the
   old 1×5 strip render (recompose like stage_C's, or let the metrics table carry E2);
   the spray localization slopes are pulled-grounder grids, not comparable to v0's, and
   that slope moved 0.0023 → 0.0010 between identical runs — do not over-read E7.
4. **Rollup A/B** (`results/rollup_ab/rollup_ab_metrics.json`): conditioning the rollup
   on per-ball spray is reliably counterproductive — marginalized beats conditioned in
   7/8 splits, paired bootstrap +0.000317 [+0.000189, +0.000458] with conditioned winning
   0 of 5,000 resamples — but the magnitude is below the 0.001 practical bar, and both
   spray rollups lose to v0 in every band (calibrated RMSE ×10⁻³, PA≥30: cond 36.59,
   marg 36.27, v0 35.82, Savant 35.56). Figure: `next_season_rmse_by_band.png`.
5. **Reading + forward pointer**: capacity dilution, not missing information — same 50
   trees spread over 5 features. `scripts/capacity_experiment.py` is written and
   smoke-tested but NOT yet run at Stage C (m_trees=200, ~5.3 h, paired per-event test);
   present it as the open experiment, not a promise. Close by handing to Part 7: while
   the surface stalled honestly, the *product* question moved — from describing a season
   to forecasting the rest of one.

## Notebook 07 — Part 7: rest-of-season forecast (talent3, rung a)

**Blocking prerequisite — resolve before writing any cells.** `results/talent3/` commits
only `NOTES.md` + four figures; `metrics.json`, the parquets, and `leakage_digest.json`
are gitignored (and absent from fresh clones), so the notebook has nothing to `jload`
and nothing to guard against. Fix on a machine that has the slim Statcast caches
(`run_talent3.py` is closed-form, ~20 s, no BART refit):

- If `results/talent3/metrics.json` is small (≲ a few hundred KB): remove its line from
  `.gitignore` and commit it.
- If it is large: extend `scripts/run_talent3.py` to also emit a slim committed
  `results/talent3/metrics_summary.json` (gate verdicts + CIs, pooled and per-k RMSE for
  model + all five benchmarks, the coverage-by-k table, n counts), re-run, commit that.

Do NOT write 07 on hardcoded, unguardable numbers.

Suggested beats (numbers per RESULTS.md / NOTES.md — re-verify against the committed
artifact once it exists):

1. **A different product**: not "how good was this season" but "stand at a hitter's
   first *k* PAs and forecast his final full-season xwOBA, with a calibrated range".
   The model in one line: `θ_{i,t} = μ_t + η_i + u_{i,t}` — the new lever over Parts 4–5
   is the **career random intercept `η_i`** from the player's own completed prior
   seasons. Closed-form Gaussian, ~20 s, reuses talent2's `bootstrap_S` and player_ci's
   forward-bootstrap idea. 7,493 forecasts, 1,945 (batter, season) pairs, 2022–2025,
   k ∈ {50,100,150,200,300}; the `assert_causal` leakage guard checked 4,674,004
   conditioning rows.
2. **The race** (`figures/rmse_vs_benchmarks.png`): pooled RMSE **0.02203** vs naive
   0.03438, league-shrunk 0.02448, Marcel 0.02270, single-season Level 2 0.02450,
   savant-to-date 0.03438 (identical to naive by construction — say why).
3. **Gates, including the failure**: G1 beats naive at k≤100 (+0.01786, CI excludes 0);
   G2 beats single-season L2 (+0.00246 [+0.00169, +0.00335] — the multi-year lever
   pays); G3 beats Marcel (+0.00067 [+0.00013, +0.00117]); **G4 calibration FAIL** —
   50%/80% central intervals run narrow (worst at high k, e.g. 50% covers 0.427 at
   k=300) while the 90% band holds within ±5pp; G5 reduces to Phase 1 exactly
   (max|Δ| 5.6e-17). Show `figures/calibration_by_band.png` and treat G4 the way Part 5
   treats its bootstrap CI — a real, open failure, not a footnote.
4. **The product shot**: `figures/fan_chart_examples.png` (and
   `figures/width_vs_pa_and_w.png` for how the band tightens as the season resolves).
5. **Close the series**: rung (b)+ — aging, peripherals in the intercept, the BART
   surface term for coverage — and the standing spray/capacity question from Part 6.

## Also update, same commit

- `notebooks/README.md`: add 06/07 rows to the table, extend the arc paragraph, keep the
  guard-cell note accurate.
- Root `README.md` §"The notebooks": arcs 6–7 are currently marked "not yet
  notebook-ized" — update that sentence.

## Definition of done

- Both notebooks execute headless from a fresh-clone artifact state (07 only after its
  prerequisite lands), zero error outputs, guard cells print their pass line.
- Figures display with captions at sane sizes; no polars dtype rows anywhere.
- Both READMEs updated; everything committed with outputs embedded and pushed.
