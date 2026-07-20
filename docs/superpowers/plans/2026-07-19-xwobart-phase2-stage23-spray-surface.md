# xwobart Phase 2 / Stages 2+3 — Spray Cache Rebuild + 5-Feature BART Surface Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the outcome surface *where the ball went*. Rebuild the slim Statcast caches with
`hc_x`/`hc_y`/`stand`, derive a pull-relative spray angle, and refit the v0 BART surface once with
five features instead of three — decisively beating the frozen holdout ELPD anchor of **−80107 ± 244**
and persisting ~200 thinned per-event value draws so Stage 4 can finally fold surface uncertainty into
player intervals.

**Architecture:** Two stages in one document because the rebuild's only consumer is the refit.
**Stage 2** (minutes) adds three columns to `KEEP_COLUMNS`, derives
`spray_pull = ±atan2(hc_x − 125.42, 198.27 − hc_y)` mirrored by `stand` so positive always means
*pulled*, and keeps `stand` as a separate feature so BART can still recover raw direction (park
asymmetries act on raw direction; batter skill acts pull-relative). It is guarded by two independent
gate families: a **reproduction gate** proving the rebuild changed nothing that already existed, and a
**sign-QC gate** proving the mirror is not backwards. **Stage 3** (~30 min) runs the *same*
orchestrator, subsample protocol and holdout as v0 Stage C — parameterized by a `--variant` flag
rather than forked into a second script, because "identical protocol" is best guaranteed by identical
code — then scores ELPD against the anchor, persists per-event value draws, and runs the
spray-conditioned vs spray-marginalized rollup A/B.

**Tech Stack:** Python 3.12, Polars (pandas only at boundaries), NumPy, pymc 6.1.0 / pymc-bart 0.12.0,
arviz, matplotlib, pytest. Statcast source via env `STATCAST_PATH` and `-e ../kinferencetoolkit`
(`pipeline.statcast_loader`, `pipeline.player_names`).

---

## Where this sits

Phase 2 Stage 1 shipped (`origin/main` @ `a16e57b`): `src/talent2.py` + `scripts/run_talent2.py`, a
joint-MVN Level-2 talent model. It needed no cache rebuild and no BART refit. Stages 2+3 need both.
Stage 4 — folding the persisted draw variance into `S_i[0,0]`, pushing the winning rollup through
Level 2, and testing 50/80/90% interval coverage — is **out of scope** and gets its own plan once
Stage 3's numbers exist.

Design of record: `docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md`
(§"Recommended spec", §"Risks" 2–3, housekeeping item 6).
Quality bar and template: `docs/superpowers/plans/2026-07-19-xwobart-phase2-level2-talent.md`.

**Skills:** @superpowers-extended-cc:test-driven-development for every TDD task;
@superpowers-extended-cc:verification-before-completion before claiming any task done;
@superpowers-extended-cc:systematic-debugging when anything fails.

**Worktree note:** work happens directly on `main`, no worktree. Commit per task; push only when asked.
Everything runs from the repo root with `.venv/bin/...`.

---

## RUNTIME WARNING — read before sequencing anything

Stage 3's fit is **~27 minutes** (v0 Stage C: 1601 s fit, 1819 s total), and the spray-marginalization
pass adds **~30 minutes** on top. Stage 1 was 7 seconds. Every Stage 2 task is seconds-to-minutes.

**Task 8 runs the fit and the marginalization in ONE ~60-minute invocation.** This is deliberate: the
marginalized values require predicting from the fitted trees, which live on the in-memory `model`
object (`model["mu"].owner.op.all_trees`) and **cannot be recovered from `idata.nc` alone**. Splitting
them across two commands would mean refitting. Task 7 wires both; Task 8 pays for them once; Task 9 is
pure analysis over saved arrays and needs no model.

**Tasks 1–7 MUST fully pass before Task 8 runs.** A mirrored-spray bug is invisible downstream — it
silently reflects half the league — and would cost an hour plus a poisoned anchor. The sign-QC gate
(Task 5) and the ~2-minute Stage-A smoke of the full wiring (Task 7 Step 7.4) both come first. Do not
reorder.

`scripts/run_v0.py` refuses fits estimated over 30 minutes without `--acknowledge-runtime`
(`run_v0.py:109`). v0's own Stage-C estimate came in at **28.46 min**, just under the bar, so the
guard may well not fire — Task 8 passes the flag anyway so the run cannot stall waiting on it.

---

## Decisions taken (confirmed with the user 2026-07-19)

1. **The rebuild is gated on byte-level reproduction of the pre-existing columns.** The real risk is
   not the added columns — `.select(KEEP_COLUMNS)` provably cannot change rows. It is that
   `build_season_caches(force=True)` re-reads the *upstream monthly cache*, which may have been
   updated by KIT since the slim caches were built. That would silently move 2,636 player-seasons,
   r 0.4886/0.4669 and ELPD −80107 with no code change at all. Gates R1/R2 catch it for ~1 minute of
   compute.
2. **`scripts/run_talent2.py` is re-run in Stage 2, not Stage 3.** It reads the slim caches' *public*
   `estimated_woba_using_speedangle` and never touches the BART surface (`run_talent2.py:47`,
   `src/talent2.py:build_pa_measurements`), so the Stage-3 refit provably cannot move its numbers —
   re-running there is a no-op. The cache rebuild *can* move them, so the re-run belongs to Stage 2's
   reproduction gate. Stage 4 is where Level 2 first genuinely consumes the surface.
3. **hc missingness is imputed and flagged, but the flag is NOT a BART feature.** Measured
   missingness on BBE is 0.0345% / 0.0434% / 0.0375% / 0.0361% for 2022–25 (≈40–55 rows a season),
   not structural by launch angle. At that rate a missing-flag feature is pure split noise; keeping
   exactly five features preserves the design's stated feature set and the anchor comparison. Rows
   are **imputed, never dropped** — dropping would move the holdout event count off 122,006, which is
   what the −80107 anchor is defined over.

---

## Frozen anchors (measured 2026-07-19; every gate below is calibrated against these)

| Quantity | Value | Source |
|---|---|---|
| Holdout ELPD (lppd) | **−80107.495 ± 243.608** over **122,006** events | `results/stage_C/metrics.json` |
| Stage-C fit rows / predict rows | 100,000 / train 363,595, holdout 122,006 | same |
| Stage-C fit runtime | 1601 s fit, 1819 s total | same |
| Weighted holdout ECE | 0.042233 | same |
| Per-class Brier — triple / HR / double | 0.005013 / 0.025344 / 0.052073 | same |
| Grounder sprint slope (per ft/s) | 0.0023488 (barrel 0.0008973) | same |
| Event replication r (holdout) | 0.9105 | same |
| BBE per season after `filter_bbe` | 2022 **118,891** · 2023 **122,070** · 2024 **122,634** · 2025 **122,006** | probe, 2026-07-19 |
| Phase-1 talent r | PA≥100 **0.4886**, PA≥30 **0.4669**, 2,636 rows | `results/talent/talent_metrics.json` |
| Level-2 talent r | PA≥100 **0.4908**, PA≥30 **0.4698** | `results/talent2/talent2_metrics.json` |
| Existing test count | **48 passing** | `.venv/bin/pytest` |

Note 118,891 + 122,070 + 122,634 = **363,595** and 2025 = **122,006** — these reproduce v0's
`predict_rows` exactly, which is why the BBE-count gate (R5) is a valid check on anchor comparability.

## Spray-angle empirical anchors (measured 2026-07-19 on the upstream cache, all four seasons)

With `φ_raw = atan2(hc_x − 125.42, 198.27 − hc_y)` in degrees — **negative = left field, positive =
right field** — and `spray_pull = −φ_raw` for `stand == "R"`, `+φ_raw` for `stand == "L"`:

| Season | mean `spray_pull` L / R | HR mean L / R | HR frac pull-side L / R | hc missing | \|φ_raw\| > 45° |
|---|---|---|---|---|---|
| 2022 | +6.84 / +3.23 | +19.3 / +17.5 | 0.819 / 0.789 | 0.0345% | 9.22% |
| 2023 | +7.36 / +3.62 | +19.2 / +16.3 | 0.808 / 0.779 | 0.0434% | 9.08% |
| 2024 | +7.15 / +3.55 | +20.0 / +18.0 | 0.827 / 0.805 | 0.0375% | 8.98% |
| 2025 | +7.50 / +3.56 | +20.3 / +17.5 | 0.836 / 0.799 | 0.0361% | 9.09% |

Named 2024 anchors (≥250 BBE): Kyle Schwarber (id 656941, L) **+13.90**; Isaac Paredes
(id 670623, R) **+11.88**; Carlos Santana (467793, L) **+13.18**; Ryan Mountcastle (663624, R)
**−5.16** (a genuine opposite-field hitter); Ke'Bryan Hayes (663647, R) **−4.25**.

`stand` is per-event: **65 of 647** batter-seasons with BBE in 2024 carry both `stand` values
(41 with ≥25 BBE from each side). It must be carried per row, never joined per player-season.

`|φ_raw| > 45°` on ~9% of BBE is expected and correct — caught fouls and pop-ups land outside the
foul lines in hc coordinates. **Do not clamp.** BART splits on it happily.

---

## Success gates

### Stage 2 — reproduction (Task 4)

| Gate | Criterion | Type |
|---|---|---|
| **R1** | Per-season row counts in the rebuilt slim caches are **identical** to the pre-rebuild counts | HARD |
| **R2** | Order-independent content digest over the **pre-existing 15 columns** is identical, per season | HARD |
| **R3** | New columns land: `stand` 100% non-null on BBE; `hc_x`/`hc_y` null rate < 0.1% per season | HARD |
| **R4** | Phase-1 talent table re-derived in memory from the new caches matches the frozen parquet: same height (2,636), `max|Δ xwoba_talent| < 1e-12` | HARD |
| **R5** | `filter_bbe` counts per season are 118,891 / 122,070 / 122,634 / 122,006 exactly (⇒ v0's 363,595 + 122,006 hold, so the ELPD anchor stays comparable) | HARD |
| **R6** | `run_talent2.py --stage full` re-run: its own G1–G4 pass **and** PA≥30 r = 0.469817, PA≥100 r = 0.490783 to within 5e-4 | HARD |

If any of R1/R2/R5 fails, the **upstream data moved**, not the code. Stop, report, and do not proceed
to Stage 3 — every frozen anchor in this repo is invalid until that is reconciled. The pre-rebuild
backup in `data/raw/prerebuild/` is the restore path; the runner prints the exact command.

### Stage 2 — spray sign QC (Task 5). Cheap, and it runs BEFORE the 27-minute fit.

| Gate | Criterion | Type |
|---|---|---|
| **S1** | League mean `spray_pull` > **+1.0°** for **both** `stand` values, in **every** season (observed min +3.23) | HARD |
| **S2** | On home runs: mean `spray_pull` ≥ **+12°** for both stands, every season (observed min +16.3) | HARD |
| **S3** | On home runs: fraction with `spray_pull > 0` ≥ **0.70** for both stands, every season (observed min 0.779) | HARD |
| **S4** | 2024 named anchors: batter 656941 (L) > +8, batter 670623 (R) > +8, batter 663624 (R) < 0 | HARD |
| **S5** | Restricted to the ~65 switch-hitter batter-seasons in 2024 (two distinct `stand` values), mean `spray_pull` > 0 **within each stand subgroup separately** | HARD |
| **S6** | `hc_imputed` rate < 0.1% per season (the "no nulls" half is asserted inside `add_spray`, so it can never reach this gate) | HARD |

**Why these families and not one "average pull is positive" check.** Enumerate the mirror bugs:
mirror *neither* hand → R mean −3.55, L +7.15 (S1 fails on R); mirror *both* → R +3.55, L −7.15 (S1
fails on L); mirror the *wrong* hand → both negative (S1 fails on both); swap the `atan2` arguments →
`atan2(Δy, Δx) = 90° − atan2(Δx, Δy)`, so RHB mean pull becomes ≈ −93.5° (S1 fails catastrophically
on R, while L reads ≈ +82.9° and passes — which is exactly why S1 requires *both* hands).

**S5 is the non-obvious one and it must be written as specified above.** The tempting version —
"count batter-seasons with two `stand` values" — is *vacuous*: that count is a property of the input
data that no bug in `add_spray` can change, and R3 already guarantees it. The failure it has to catch
is resolving `stand` per *player-season* (modal hand) instead of per event. That flips the sign only
on the minority-hand events of 65 of 647 batter-seasons — league means move by well under 0.1°, so
S1/S2/S3 all pass, and none of the three named anchors is a switch hitter, so S4 passes too. Splitting
the switch-hitter rows by `stand` and requiring each subgroup to lean positive is what makes the
minority-hand flip visible. S4 remains the human-legible confirmation that survives a reviewer's
skepticism.

**This was verified by simulating the bug on 2024 data (2026-07-19), not reasoned about:**

| | league mean L / R | HR mean L / R | HR frac L / R | switch-only mean L / R |
|---|---|---|---|---|
| correct per-event mirror | +7.146 / +3.549 | +20.0 / +18.0 | .827 / .805 | **+8.679 / +5.603** |
| modal-hand mirror (the bug) | +7.118 / +2.991 | +20.0 / +15.8 | .827 / .765 | **+8.520 / −3.969** |
| gate outcome under the bug | S1 **PASS** | S2 **PASS** | S3 **PASS** | S5 **FAIL** |

S5 is the only gate that catches it, and it catches it decisively (−3.969 against a > 0 threshold,
on 4,136 rows).

### Stage 3 — the surface (Tasks 8–9)

| Gate | Criterion | Type |
|---|---|---|
| **E1** | Holdout ELPD (lppd) beats **−80107.495 by ≥ 1000 nats** (≈ +0.008 nats/event; also ≥ 3× the 243.6 anchor SE) | HARD — the thesis |
| **E2** | Weighted holdout ECE ≤ **0.0465** (v0's 0.042233 + 10%) — the new surface must not buy ELPD with calibration | HARD |
| **E3** | Per-class Brier for **triple** ≤ v0's 0.005013; spray should *help* the rare class (triples are geometrically concentrated) | DIAGNOSTIC |
| **E4** | `verify_oos_mechanism` passes (corr > 0.99, mean abs diff < 0.03). BART `mu` R-hat is structurally high and is **NOT** a convergence signal | HARD |
| **E5** | No collapsed class probabilities (`sanity_check`) | HARD |
| **E6** | Per-event value draws persisted: 200 draws × 363,595 train + 200 × 122,006 holdout, float32, row-aligned key parquets written and asserted | HARD |
| **E7** | Variable importance ranks `spray_pull` in the top 3 — i.e. **column index 2 appears in `variable_importance.raw.indices[:3]`** (v0's value is `[1, 0, 2]`, a ranking of *column indices*, and the whole dict degrades to `{"unavailable": ...}` on any exception, so check for that first); sprint speed's grounder slope migrates — pulled-grounder slope > oppo-grounder slope on the contact grid; the LA×spray HR-band PDP peaks on the pulled side | DIAGNOSTIC |
| **E8** | Rollup A/B reported both ways with next-season RMSE + r by PA band and a paired bootstrap. **The conditioned rollup is not assumed to win** | PROTOCOL |

**If E1 fails — stop and report.** The design says a non-beat is a red flag, not a tuning
opportunity. Write every metric, then work this checklist in order before touching any
hyperparameter: (1) is `spray_pull` actually in `X_fit` — print `X_fit[:3]` and the feature list;
(2) did the sign QC run against the *rebuilt* caches or a stale frame; (3) is `stand_R` constant
(a broken cast makes it 0.0 everywhere); (4) is the holdout event count still 122,006 — a changed
denominator makes the ELPD incomparable, not worse; (5) is the subsample still 100,000 rows at
seed 42. Do **not** raise `m_trees`, `draws`, or the subsample to rescue it — that breaks anchor
comparability, which is the whole point of reusing the protocol.

---

## File structure

| File | Responsibility |
|---|---|
| `src/prep.py` | **Modify.** Add `HOME_PLATE_X/Y`, `_spray_cols`, `spray_impute_table`, `add_spray`; split `FEATURES` into `FEATURES_V0` / `FEATURES_SPRAY`; parameterize `build_features` |
| `src/data.py` | **Modify.** Add `hc_x`, `hc_y`, `stand` to `KEEP_COLUMNS`; add `cache_fingerprint` |
| `src/evaluate.py` | **Modify.** Parameterize `variable_importance(labels=…)` and `contact_grids(variant=…)`; add `SPRAY_PULL_DEG` and `la_spray_grid` (the spec's LA×spray HR-band PDP) |
| `scripts/rebuild_caches.py` | **Create.** Backup → rebuild → gates R1–R6. The only script that ever passes `force=True` |
| `scripts/qc_spray.py` | **Create.** Gates S1–S6 + two figures. Seconds to run; runs before the fit |
| `scripts/run_v0.py` | **Modify.** `--variant {v0,spray}`, `--persist-draws N`, `--marginalize-spray M`; variant-aware stage dir, features, grids, importance labels; LA×spray HR-band PDP figure; ELPD durability write; draw persistence; spray marginalization; best-effort `all_trees` pickle (all three must run while the model is live, and after the ELPD verdict is on disk) |
| `scripts/rollup_ab.py` | **Create.** Pure analysis over the saved arrays: two rollups, the A/B race against next-season actual wOBA, the descriptive comparison. No model, no refit |
| `tests/test_prep.py` | **Modify.** +6 spray/feature tests |
| `tests/test_data.py` | **Modify.** +4 KEEP_COLUMNS/fingerprint tests |
| `tests/test_evaluate.py` | **Modify.** +3 variant-grid tests |
| `.gitignore` | **Modify.** `results/**/*.npy`, `results/**/*.pkl`, `results/**/ev_draws_keys_*.parquet` — ~390 MB of draws, the trees pickle, and ~13 MB of key parquets |
| `results/stage2_rebuild/` (generated) | `rebuild_report.json`, `spray_qc.json`, `figures/spray_*.png`, `NOTES.md` |
| `results/stage_C_spray/` (generated) | `metrics.json`, `idata.nc`, `player_table.parquet`, `figures/` (incl. `pdp_la_spray_hr.png`), `ev_draws_{train,holdout}.npy`, `ev_marginalized_{train,holdout}.npy`, `ev_draws_keys_{train,holdout}.parquet`, `lppd_i_holdout.npy`, `all_trees.pkl` |
| `results/rollup_ab/` (generated) | `rollup_ab_metrics.json`, `marginalized_values.parquet`, `figures/`, `NOTES.md` |
| `results/RESULTS.md` | **Modify.** New "Phase 2 Stage 3 — spray surface" section + deviations |

Test count: **48 → 61**.

---

### Task 1: Spray angle derivation + imputation table (TDD)

**Files:**
- Modify: `src/prep.py`
- Test: `tests/test_prep.py`

The whole stage rests on this function's sign convention. It gets the most explicit tests in the plan.

- [ ] **Step 1.1: Write the failing tests (append to `tests/test_prep.py`)**

```python
from src.prep import HOME_PLATE_X, HOME_PLATE_Y, add_spray, spray_impute_table


def _spray_df():
    """Four BBE with hand-computed geometry, plus one with hc missing.

    hc_x < 125.42 is LEFT field, hc_x > 125.42 is RIGHT field; hc_y DECREASES going
    out toward the outfield, so 198.27 - hc_y > 0 for any ball in play.
    """
    return pl.DataFrame({
        "batter":       [1, 2, 3, 4, 5],
        "game_year":    [2024] * 5,
        "stand":        ["R", "L", "R", "L", "R"],
        # 45 deg to the LEFT, 45 deg to the RIGHT, dead center, 45 left, missing
        "hc_x":         [75.42, 175.42, 125.42, 75.42, None],
        "hc_y":         [148.27, 148.27, 148.27, 148.27, None],
        "launch_speed": [100.0, 100.0, 100.0, 100.0, 100.0],
        "launch_angle": [25.0, 25.0, 25.0, 25.0, 25.0],
    })


def test_add_spray_sign_convention_is_pull_relative():
    cell, hand = spray_impute_table(_spray_df())
    out = add_spray(_spray_df(), cell, hand).sort("batter")
    # raw direction: negative = left field, positive = right field
    assert abs(out["phi_raw"][0] - (-45.0)) < 1e-9      # RHB pulled to left
    assert abs(out["phi_raw"][1] - (+45.0)) < 1e-9      # LHB pulled to right
    assert abs(out["phi_raw"][2] - 0.0) < 1e-9          # dead center
    assert abs(out["phi_raw"][3] - (-45.0)) < 1e-9      # LHB went OPPOSITE field
    # pull-relative: POSITIVE means pulled, for BOTH hands
    assert abs(out["spray_pull"][0] - (+45.0)) < 1e-9   # RHB to left  = pulled
    assert abs(out["spray_pull"][1] - (+45.0)) < 1e-9   # LHB to right = pulled
    assert abs(out["spray_pull"][2] - 0.0) < 1e-9
    assert abs(out["spray_pull"][3] - (-45.0)) < 1e-9   # LHB to left  = opposite
    assert out["stand_R"].to_list() == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_add_spray_mirror_is_not_identity_or_global_flip():
    """Guards the two mirror bugs a single sign test cannot distinguish."""
    out = add_spray(_spray_df(), *spray_impute_table(_spray_df())).sort("batter")
    phi, sp = out["phi_raw"].to_numpy()[:4], out["spray_pull"].to_numpy()[:4]
    assert not np.allclose(sp, phi)          # not "forgot to mirror"
    assert not np.allclose(sp, -phi)         # not "mirrored both hands"


def test_add_spray_imputes_missing_hc_and_flags_without_dropping():
    df = _spray_df()
    cell, hand = spray_impute_table(df)
    out = add_spray(df, cell, hand).sort("batter")
    assert out.height == df.height                      # NEVER drop: ELPD anchor n
    assert out["spray_pull"].null_count() == 0
    assert out["hc_imputed"].to_list() == [False, False, False, False, True]
    # RHB with no hc falls back to the RHB median (cells need >= 25 rows, so the
    # per-hand fallback is what fires here): median of the two RHB spray values
    assert abs(out["spray_pull"][4] - float(np.median([45.0, 0.0]))) < 1e-9


def test_spray_impute_table_uses_cells_when_dense_enough():
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame({
        "batter": list(range(n)), "game_year": [2024] * n, "stand": ["R"] * n,
        # all in one (ev_bin, la_bin) cell, all pulled to left field
        "hc_x": (125.42 - rng.uniform(20, 60, n)).tolist(),
        "hc_y": [148.27] * n,
        "launch_speed": [101.0] * n, "launch_angle": [26.0] * n,
    })
    cell, hand = spray_impute_table(df)
    assert cell.height == 1 and cell["spray_cell"][0] > 0     # pulled, positive
    miss = df.head(1).with_columns(hc_x=pl.lit(None, dtype=pl.Float64),
                                   hc_y=pl.lit(None, dtype=pl.Float64))
    out = add_spray(miss, cell, hand)
    assert out["hc_imputed"][0] and abs(out["spray_pull"][0] - cell["spray_cell"][0]) < 1e-9


def test_add_spray_rejects_null_stand():
    bad = _spray_df().with_columns(stand=pl.lit(None, dtype=pl.Utf8))
    with pytest.raises(AssertionError, match="stand"):
        add_spray(bad, *spray_impute_table(_spray_df()))
```

Add `import pytest` at the top of `tests/test_prep.py` if it is not already there (it is not — the
file currently imports only `polars` and `numpy`).

- [ ] **Step 1.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_prep.py -v`
Expected: collection of the whole file FAILS with
`ImportError: cannot import name 'HOME_PLATE_X' from 'src.prep'` — a module-level import error
errors out all 12 tests in the file, not just the 5 new ones.

- [ ] **Step 1.3: Implement (append to `src/prep.py`, after `build_features`)**

```python
# Statcast hit-coordinate origin (home plate). hc_x increases toward RIGHT field;
# hc_y DECREASES going out toward the outfield, hence the (198.27 - hc_y) term.
HOME_PLATE_X = 125.42
HOME_PLATE_Y = 198.27
SPRAY_EV_BIN = 5.0        # mph, for the imputation lookup
SPRAY_LA_BIN = 10.0       # degrees
SPRAY_MIN_CELL = 25       # rows required before an (ev, la, stand) cell is trusted


def _spray_cols(bbe: pl.DataFrame) -> pl.DataFrame:
    """phi_raw (RAW direction, degrees: negative = left field, positive = right),
    stand_R (1.0 / 0.0), the observed pull-relative angle, and the lookup bins.

    A right-handed batter PULLS to left field, so the pull-relative angle negates
    phi_raw for stand == 'R' and leaves stand == 'L' alone. Verified empirically on
    2022-25: league mean pull is positive for BOTH hands (L +6.8..+7.5, R +3.2..+3.6)
    and home runs sit at +16..+20 for both, ~80% on the pull side."""
    assert bbe["stand"].null_count() == 0, "stand must be non-null (it is per-EVENT)"
    return bbe.with_columns(
        stand_R=(pl.col("stand") == "R").cast(pl.Float64),
        phi_raw=pl.arctan2(pl.col("hc_x") - HOME_PLATE_X,
                           HOME_PLATE_Y - pl.col("hc_y")).degrees(),
    ).with_columns(
        spray_obs=pl.when(pl.col("stand") == "R").then(-pl.col("phi_raw"))
                    .otherwise(pl.col("phi_raw")),
        _ev_bin=(pl.col("launch_speed") // SPRAY_EV_BIN).cast(pl.Int32),
        _la_bin=(pl.col("launch_angle") // SPRAY_LA_BIN).cast(pl.Int32),
    )


def spray_impute_table(bbe: pl.DataFrame) -> tuple[pl.DataFrame, dict[float, float]]:
    """Median pull-relative spray by (ev_bin, la_bin, stand_R), plus a per-hand
    fallback. Build this on the TRAINING seasons only and apply it to both train and
    holdout, so the holdout never imputes itself."""
    d = _spray_cols(bbe).drop_nulls("spray_obs")
    cell = (
        d.group_by("_ev_bin", "_la_bin", "stand_R")
        .agg(spray_cell=pl.col("spray_obs").median(), _n=pl.len())
        .filter(pl.col("_n") >= SPRAY_MIN_CELL)
        .drop("_n")
        .sort("_ev_bin", "_la_bin", "stand_R")     # polars group_by is not order-stable
    )
    hand = {float(k): float(v) for k, v in
            d.group_by("stand_R").agg(m=pl.col("spray_obs").median()).iter_rows()}
    return cell, hand


def add_spray(bbe: pl.DataFrame, cell: pl.DataFrame,
              hand: dict[float, float]) -> pl.DataFrame:
    """Add phi_raw, spray_pull (POSITIVE = pulled, both hands), stand_R, hc_imputed.

    Rows with null hc_x/hc_y (0.034-0.043% of BBE, 2022-25) are IMPUTED, never dropped
    -- dropping would move the holdout event count off 122,006 and make the -80107 ELPD
    anchor incomparable. Fallback ladder: (ev, la, stand) cell median -> per-hand median
    -> 0.0. hc_imputed is an AUDIT column, deliberately not a BART feature: at ~45 rows
    a season a flag feature is split noise, and five features is the design's spec."""
    d = _spray_cols(bbe).join(cell, on=["_ev_bin", "_la_bin", "stand_R"], how="left")
    fallback = (pl.when(pl.col("stand_R") == 1.0).then(pl.lit(hand.get(1.0, 0.0)))
                  .otherwise(pl.lit(hand.get(0.0, 0.0))))
    out = d.with_columns(
        hc_imputed=pl.col("spray_obs").is_null(),
        spray_pull=pl.coalesce(pl.col("spray_obs"), pl.col("spray_cell"), fallback),
    ).drop("spray_obs", "spray_cell", "_ev_bin", "_la_bin")
    assert out["spray_pull"].null_count() == 0, "spray_pull must be fully imputed"
    return out
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prep.py -v`
Expected: all pass (7 existing + 5 new)

- [ ] **Step 1.5: Full suite, then commit**

Run: `.venv/bin/pytest` — Expected: **53 passed**

```bash
git add src/prep.py tests/test_prep.py
git commit -m "feat(prep): pull-relative spray angle with stand mirror, cell imputation, audit flag"
```

---

### Task 2: Variant-aware feature list (TDD)

**Files:**
- Modify: `src/prep.py`
- Test: `tests/test_prep.py`

`FEATURES` becomes two named lists. `build_features` takes the list so one orchestrator can serve both
variants — the cheapest possible guarantee that the spray fit uses v0's exact protocol.

- [ ] **Step 2.1: Write the failing test (append)**

```python
from src.prep import FEATURES_SPRAY, FEATURES_V0


def test_feature_lists_and_variant_selection():
    assert FEATURES_V0 == ["launch_speed", "launch_angle", "sprint_speed"]
    assert FEATURES_SPRAY == ["launch_speed", "launch_angle", "spray_pull",
                              "stand_R", "sprint_speed"]
    df = pl.DataFrame({
        "launch_speed": [90.0, 100.0], "launch_angle": [10.0, 25.0],
        "spray_pull": [-12.0, 30.0], "stand_R": [1.0, 0.0],
        "sprint_speed": [27.0, 29.5], "outcome_class": [0, 4], "extra": ["a", "b"],
    })
    X3, y = build_features(df)                       # default stays v0: 3 columns
    assert X3.shape == (2, 3) and X3[1, 2] == 29.5
    X5, _ = build_features(df, FEATURES_SPRAY)
    assert X5.shape == (2, 5) and X5.dtype == np.float64
    assert X5[0].tolist() == [90.0, 10.0, -12.0, 1.0, 27.0]
    assert y.tolist() == [0, 4]
```

- [ ] **Step 2.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_prep.py::test_feature_lists_and_variant_selection -v`
Expected: FAIL — `ImportError: cannot import name 'FEATURES_SPRAY'`

- [ ] **Step 2.3: Implement — replace `src/prep.py:10` and `build_features`**

Replace line 10:
```python
FEATURES_V0 = ["launch_speed", "launch_angle", "sprint_speed"]
FEATURES_SPRAY = ["launch_speed", "launch_angle", "spray_pull", "stand_R", "sprint_speed"]
FEATURES = FEATURES_V0          # back-compat default; run_v0 --variant selects
VARIANT_FEATURES = {"v0": FEATURES_V0, "spray": FEATURES_SPRAY}
```

Replace `build_features`:
```python
def build_features(bbe: pl.DataFrame, features: list[str] | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix, float64; no standardization (BART does not need it). `features`
    defaults to the v0 three so existing callers and the frozen v0 path are unchanged."""
    cols = features or FEATURES_V0
    X = bbe.select(cols).to_numpy().astype(np.float64)
    y = bbe["outcome_class"].to_numpy().astype(np.int64)
    return X, y
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prep.py -v` — Expected: all pass
Run: `.venv/bin/pytest` — Expected: **54 passed**

- [ ] **Step 2.5: Commit**

```bash
git add src/prep.py tests/test_prep.py
git commit -m "feat(prep): variant-aware FEATURES lists; build_features takes the column list"
```

---

### Task 3: KEEP_COLUMNS + cache fingerprint (TDD)

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 3.1: Write the failing tests (append to `tests/test_data.py`)**

```python
from src.data import KEEP_COLUMNS, cache_fingerprint

PRE_REBUILD_COLUMNS = [
    "game_pk", "game_date", "game_year", "batter", "events", "description",
    "des", "type", "bb_type", "launch_speed", "launch_angle",
    "launch_speed_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
]


def test_keep_columns_adds_spray_inputs_without_removing_anything():
    assert KEEP_COLUMNS[:len(PRE_REBUILD_COLUMNS)] == PRE_REBUILD_COLUMNS
    assert KEEP_COLUMNS[len(PRE_REBUILD_COLUMNS):] == ["hc_x", "hc_y", "stand"]


def _fp_frame(tmp_path, extra=None, vals=(1, 2, 3)):
    d = {"a": list(vals), "b": ["x", "y", "z"]}
    if extra:
        d |= extra
    # The filename MUST discriminate on content: two frames differing only in `vals`
    # would otherwise collide and silently overwrite each other, making the
    # changed-value assertion below compare a file to itself and always fail.
    p = tmp_path / f"f{len(d)}_{'-'.join(map(str, vals))}.parquet"
    pl.DataFrame(d).write_parquet(p)
    return p


def test_fingerprint_ignores_added_columns(tmp_path):
    p1 = _fp_frame(tmp_path)
    p2 = _fp_frame(tmp_path, extra={"c": [9.0, 9.0, 9.0]})
    assert cache_fingerprint(p1, ["a", "b"]) == cache_fingerprint(p2, ["a", "b"])


def test_fingerprint_ignores_row_order(tmp_path):
    p1 = _fp_frame(tmp_path, vals=(1, 2, 3))
    p2 = tmp_path / "rev.parquet"
    pl.read_parquet(p1).reverse().write_parquet(p2)
    assert cache_fingerprint(p1, ["a", "b"]) == cache_fingerprint(p2, ["a", "b"])


def test_fingerprint_detects_a_changed_value_and_a_changed_row_count(tmp_path):
    p1 = _fp_frame(tmp_path, vals=(1, 2, 3))
    p2 = _fp_frame(tmp_path, vals=(1, 2, 4))
    assert cache_fingerprint(p1, ["a", "b"])["digest"] != cache_fingerprint(p2, ["a", "b"])["digest"]
    p3 = tmp_path / "short.parquet"
    pl.read_parquet(p1).head(2).write_parquet(p3)
    assert cache_fingerprint(p3, ["a", "b"])["n_rows"] == 2
```

- [ ] **Step 3.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: collection of the whole file FAILS with
`ImportError: cannot import name 'cache_fingerprint'` (all 10 tests in the file error, not just the 4 new).

- [ ] **Step 3.3: Implement**

In `src/data.py`, add `import hashlib` to the imports, extend `KEEP_COLUMNS` (append only — leaving
the existing 15 in place and in order is what makes gate R2 meaningful):

```python
KEEP_COLUMNS = [
    "game_pk", "game_date", "game_year", "batter", "events", "description",
    "des", "type", "bb_type", "launch_speed", "launch_angle",
    "launch_speed_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
    # Phase 2 Stage 2: spray angle inputs. hc_x/hc_y are the hit coordinates;
    # stand is the batter's side FOR THAT EVENT (switch hitters change it mid-season).
    "hc_x", "hc_y", "stand",
]
```

and append:

```python
def cache_fingerprint(path: Path, columns: list[str]) -> dict:
    """Row count + an order-independent content digest over `columns` in one slim cache.

    Proves that adding columns to KEEP_COLUMNS changed no pre-existing value (Stage-2
    gate R2). Order independence is deliberate: a pure row-order change is not a data
    change, and neither parquet IO nor polars group_by guarantees stable ordering."""
    df = pl.read_parquet(path, columns=columns)
    h = df.hash_rows(seed=0).sort()
    return {"n_rows": df.height,
            "digest": hashlib.sha256(h.to_numpy().tobytes()).hexdigest()[:16]}
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data.py -v` — Expected: all pass
Run: `.venv/bin/pytest` — Expected: **58 passed**

- [ ] **Step 3.5: Commit** (no rebuild yet — the caches are still the old ones)

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(data): add hc_x/hc_y/stand to KEEP_COLUMNS; order-independent cache fingerprint"
```

---

### Task 4: Rebuild the caches behind the reproduction gate (R1–R6)

**Files:**
- Create: `scripts/rebuild_caches.py`

This is the only script in the repo that ever passes `force=True`. It is destructive, so it backs up
first and never auto-restores (the rebuilt files are the evidence when a gate fails).

- [ ] **Step 4.1: Implement `scripts/rebuild_caches.py`**

```python
"""Stage 2: rebuild the slim Statcast caches with hc_x/hc_y/stand, behind a hard
reproduction gate (R1-R6). Run from repo root:
    .venv/bin/python scripts/rebuild_caches.py

The gate exists because build_season_caches(force=True) re-reads the UPSTREAM monthly
cache, which KIT may have updated since these caches were built. Adding columns cannot
change existing rows; an upstream data revision silently would -- and every frozen
anchor in this repo (2,636 player-seasons; r 0.4886/0.4669; ELPD -80107 over 122,006
events) depends on those rows being unchanged.
Plan: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src import data, prep
from src.config import load_config
from src.talent import build_pa_values, build_talent_table

# The 15 columns that existed BEFORE this stage. R2 compares exactly these.
PRE_REBUILD_COLUMNS = data.KEEP_COLUMNS[:15]
# Frozen anchors (results/stage_C/metrics.json, results/talent*/; measured 2026-07-19)
BBE_PER_SEASON = {2022: 118891, 2023: 122070, 2024: 122634, 2025: 122006}
P1_ROWS = 2636
L2_R_PA30, L2_R_PA100 = 0.469817, 0.490783     # full precision: a 4-dp anchor eats
                                               # 17% of a 1e-4 tolerance on rounding alone


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} - {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    cfg = load_config()
    outdir = cfg.results_dir / "stage2_rebuild"
    outdir.mkdir(parents=True, exist_ok=True)
    backup = cfg.raw_dir / "prerebuild"
    backup.mkdir(parents=True, exist_ok=True)
    paths = {y: cfg.raw_dir / f"statcast-{y}-slim.parquet" for y in cfg.all_seasons}
    gates: list[dict] = []

    print("[1/5] fingerprinting + backing up the existing caches")
    before = {y: data.cache_fingerprint(p, PRE_REBUILD_COLUMNS) for y, p in paths.items()}
    for y, p in paths.items():
        shutil.copy2(p, backup / p.name)
    print(f"  backup -> {backup}")

    print("[2/5] rebuilding (force=True) - this re-reads the upstream monthly cache")
    data.build_season_caches(cfg, force=True)

    print("[3/5] reproduction gates")
    after = {y: data.cache_fingerprint(p, PRE_REBUILD_COLUMNS) for y, p in paths.items()}
    gates.append(_gate("R1.rows",
                       all(before[y]["n_rows"] == after[y]["n_rows"] for y in paths),
                       str({y: (before[y]["n_rows"], after[y]["n_rows"]) for y in paths})))
    gates.append(_gate("R2.digest",
                       all(before[y]["digest"] == after[y]["digest"] for y in paths),
                       str({y: (before[y]["digest"], after[y]["digest"]) for y in paths})))

    new_cols = {}
    for y, p in paths.items():
        df = pl.read_parquet(p, columns=["type", "hc_x", "hc_y", "stand"])
        x = df.filter(pl.col("type") == "X")
        new_cols[y] = {
            "stand_null_rate_bbe": float(x["stand"].is_null().mean()),
            "hc_null_rate_bbe": float(
                (x["hc_x"].is_null() | x["hc_y"].is_null()).mean()),
        }
    gates.append(_gate("R3.new_columns",
                       all(v["stand_null_rate_bbe"] == 0.0 and v["hc_null_rate_bbe"] < 0.001
                           for v in new_cols.values()), str(new_cols)))

    print("[4/5] downstream reproduction")
    pitches = pl.concat([pl.read_parquet(paths[y]) for y in cfg.all_seasons])
    p1_new = build_talent_table(build_pa_values(pitches), fit_min_pa=cfg.min_pa)
    p1_old = pl.read_parquet(cfg.results_dir / "talent" / "talent_table.parquet")
    j = p1_new.select("batter", "season", t_new="xwoba_talent").join(
        p1_old.select("batter", "season", t_old="xwoba_talent"),
        on=["batter", "season"], how="inner").sort("batter", "season")
    dmax = float((j["t_new"] - j["t_old"]).abs().max()) if j.height else float("inf")
    gates.append(_gate("R4.phase1",
                       p1_new.height == P1_ROWS and j.height == P1_ROWS and dmax < 1e-12,
                       f"rows {p1_new.height} (want {P1_ROWS}), joined {j.height}, "
                       f"max|delta| {dmax:.3e}"))

    bbe_counts = {}
    for y in cfg.all_seasons:
        b, _ = prep.filter_bbe(pl.read_parquet(paths[y]))
        bbe_counts[y] = b.height
    gates.append(_gate("R5.bbe_counts", bbe_counts == BBE_PER_SEASON,
                       f"{bbe_counts} vs frozen {BBE_PER_SEASON}"))

    # Persist R1-R5 BEFORE the Level-2 subprocess. The rebuild is already destructive by
    # this point; if step 5 raises, the evidence for the gates that DID run must survive.
    report = {"before": before, "after": after, "new_columns": new_cols,
              "bbe_counts": bbe_counts, "gates": gates}
    (outdir / "rebuild_report.json").write_text(json.dumps(report, indent=2, default=float))

    print("[5/5] re-running the Level-2 talent model against the rebuilt caches")
    root = Path(__file__).resolve().parents[1]
    rc = subprocess.run([sys.executable, "scripts/run_talent2.py", "--stage", "full"],
                        capture_output=True, text=True, cwd=root)
    print(rc.stdout[-3000:] or rc.stderr[-3000:])
    m2 = json.loads((cfg.results_dir / "talent2" / "talent2_metrics.json").read_text())
    # Key path verified against the shipped results/talent2/talent2_metrics.json:
    # l2b -> {hypers, sigma_talent_corr, pooled_pa100, pooled_pa30, by_band, split,
    #         gates, paired_bootstrap_pa30, ablations, offdiag_tripwire, ...}
    r30 = m2["l2b"]["pooled_pa30"]["xwoba_talent2"]["r"]
    r100 = m2["l2b"]["pooled_pa100"]["xwoba_talent2"]["r"]
    gates.append(_gate("R6.level2",
                       rc.returncode == 0 and abs(r30 - L2_R_PA30) < 5e-4
                       and abs(r100 - L2_R_PA100) < 5e-4,
                       f"exit {rc.returncode}, r30 {r30:.6f} (want {L2_R_PA30}), "
                       f"r100 {r100:.6f} (want {L2_R_PA100})"))

    report |= {"level2": {"r30": r30, "r100": r100}, "gates": gates}
    (outdir / "rebuild_report.json").write_text(json.dumps(report, indent=2, default=float))
    failed = [g["name"] for g in gates if not g["pass"]]
    print(f"  wrote {outdir}/rebuild_report.json")
    if failed:
        print(f"\n  HARD GATE FAILURES: {failed}")
        print("  The upstream data moved -- this is NOT a code bug. Do not proceed to")
        print("  Stage 3; every frozen anchor is invalid until reconciled. Restore with:")
        print(f"    cp {backup}/statcast-*-slim.parquet {cfg.raw_dir}/")
        raise SystemExit(1)
    print("\n  All reproduction gates PASS. Caches now carry hc_x/hc_y/stand.")


if __name__ == "__main__":
    main()
```

**The R6 key path is already verified** against the shipped `results/talent2/talent2_metrics.json`
(2026-07-19): top level is `l2a`/`l2b`, and `l2b`'s keys are `hypers, sigma_talent_corr,
pooled_pa100, pooled_pa30, by_band, split, gates, paired_bootstrap_pa30, ablations,
offdiag_tripwire, hypers_2224_sensitivity, interval_width_by_pa`. Use the lookups as written — do
not "adjust" them.

- [ ] **Step 4.2: Run the rebuild**

Run: `.venv/bin/python scripts/rebuild_caches.py`
Expected: R1–R6 all PASS, exit 0. Rebuild ~1–3 min (the upstream load dominates); the Level-2 re-run
adds ~2 min for its bootstrap.

If R1/R2/R5 fail: **stop and report.** Do not "fix" it by re-freezing the anchors.

- [ ] **Step 4.3: Commit**

`data/` is gitignored, so only the report and any Level-2 output drift are committed. Check
`git status` — if `results/talent2/` shows changes, the gates passed but the outputs are not
byte-stable (figure metadata); note that and commit them.

```bash
git add scripts/rebuild_caches.py results/stage2_rebuild/
git commit -m "feat(data): rebuild slim caches with hc_x/hc_y/stand behind reproduction gates R1-R6"
```

---

### Task 5: Spray sign QC — the gate that must precede the 27-minute fit (S1–S6)

**Files:**
- Create: `scripts/qc_spray.py`

- [ ] **Step 5.1: Implement `scripts/qc_spray.py`**

```python
"""Stage 2 sign QC: prove the pull-relative spray mirror is not backwards, BEFORE
spending 27 minutes on the surface refit. Run from repo root:
    .venv/bin/python scripts/qc_spray.py

Getting the mirror wrong silently reflects half the league and would poison the Stage-3
fit invisibly. Every single-fault mirror error (forgot to mirror / mirrored both hands /
mirrored the wrong hand / swapped the atan2 arguments) trips at least one HARD gate here.
Plan: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src import prep
from src.config import load_config

# 2024 named anchors (id, stand, comparator, threshold). Frozen IDs, not names: the
# name resolver is a network/cache dependency and must not be able to fail a gate.
NAMED_2024 = [
    (656941, "Kyle Schwarber",   "L", "gt", 8.0),    # measured +13.90
    (670623, "Isaac Paredes",    "R", "gt", 8.0),    # measured +11.88
    (663624, "Ryan Mountcastle", "R", "lt", 0.0),    # measured  -5.16 (oppo)
]
C_L, C_R = "#4878CF", "#EE854A"


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} - {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    cfg = load_config()
    outdir = cfg.results_dir / "stage2_rebuild"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    raw = {y: pl.read_parquet(cfg.raw_dir / f"statcast-{y}-slim.parquet")
           for y in cfg.all_seasons}
    bbe = {y: prep.filter_bbe(df)[0] for y, df in raw.items()}
    # Imputation table from the TRAINING seasons only; applied to every season.
    cell, hand = prep.spray_impute_table(pl.concat([bbe[y] for y in cfg.train_seasons]))
    sp = {y: prep.add_spray(b, cell, hand) for y, b in bbe.items()}

    gates: list[dict] = []
    per_season: dict[int, dict] = {}
    for y, d in sp.items():
        obs = d.filter(~pl.col("hc_imputed"))
        g = (obs.group_by("stand").agg(mean_pull=pl.col("spray_pull").mean(), n=pl.len())
                .sort("stand"))
        hr = (obs.filter(pl.col("events") == "home_run")
                 .group_by("stand").agg(mean_pull=pl.col("spray_pull").mean(),
                                        frac_pos=(pl.col("spray_pull") > 0).mean(),
                                        n=pl.len()).sort("stand"))
        per_season[y] = {
            "n_bbe": d.height,
            "hc_imputed_rate": float(d["hc_imputed"].mean()),
            "abs_phi_gt_45": float((obs["phi_raw"].abs() > 45).mean()),
            "mean_pull": dict(zip(g["stand"], g["mean_pull"])),
            "hr_mean_pull": dict(zip(hr["stand"], hr["mean_pull"])),
            "hr_frac_pull_side": dict(zip(hr["stand"], hr["frac_pos"])),
            "hr_n": dict(zip(hr["stand"], hr["n"])),
        }

    def _all(key: str, cmp) -> bool:
        return all(len(v[key]) == 2 and all(cmp(x) for x in v[key].values())
                   for v in per_season.values())

    gates.append(_gate("S1.league_mean_pull_positive", _all("mean_pull", lambda x: x > 1.0),
                       str({y: {k: round(v, 2) for k, v in d["mean_pull"].items()}
                            for y, d in per_season.items()})))
    gates.append(_gate("S2.hr_mean_pull", _all("hr_mean_pull", lambda x: x >= 12.0),
                       str({y: {k: round(v, 1) for k, v in d["hr_mean_pull"].items()}
                            for y, d in per_season.items()})))
    gates.append(_gate("S3.hr_frac_pull_side", _all("hr_frac_pull_side", lambda x: x >= 0.70),
                       str({y: {k: round(v, 3) for k, v in d["hr_frac_pull_side"].items()}
                            for y, d in per_season.items()})))

    d24 = sp[2024].filter(~pl.col("hc_imputed"))
    agg = (d24.group_by("batter", "stand").agg(mp=pl.col("spray_pull").mean(), n=pl.len())
              .filter(pl.col("n") >= 250).sort("batter", "stand"))
    named, ok_named = {}, True
    for bid, nm, st, how, thr in NAMED_2024:
        row = agg.filter((pl.col("batter") == bid) & (pl.col("stand") == st))
        val = float(row["mp"][0]) if row.height else float("nan")
        good = (val > thr) if how == "gt" else (val < thr)
        named[nm] = {"stand": st, "mean_pull": val, "want": f"{how} {thr}", "pass": bool(good)}
        ok_named &= bool(good)
    gates.append(_gate("S4.named_anchors", ok_named, json.dumps(named, default=float)))

    # S5: the mirror must be resolved PER EVENT, not per player-season. Counting switch
    # hitters would be vacuous (no bug in add_spray can change that count). Instead split
    # the switch hitters' own rows by stand: under a modal-hand mirror the minority-hand
    # subgroup flips negative while every league-level gate above still passes.
    switch_ids = (sp[2024].group_by("batter").agg(k=pl.col("stand").n_unique())
                          .filter(pl.col("k") > 1)["batter"].to_list())   # list, not Series:
                                                    # is_in on a same-dtype Series is deprecated
    sw = (d24.filter(pl.col("batter").is_in(switch_ids))
             .group_by("stand").agg(mean_pull=pl.col("spray_pull").mean(), n=pl.len())
             .sort("stand"))
    sw_ok = sw.height == 2 and bool((sw["mean_pull"] > 0).all()) and len(switch_ids) >= 1
    gates.append(_gate("S5.stand_is_per_event", sw_ok,
                       f"{len(switch_ids)} switch batters (expect ~65); by stand "
                       f"{dict(zip(sw['stand'], [round(v, 2) for v in sw['mean_pull']]))}"))
    gates.append(_gate("S6.imputation_rate",
                       all(v["hc_imputed_rate"] < 0.001 for v in per_season.values()),
                       str({y: round(v["hc_imputed_rate"] * 100, 4) for y, v in per_season.items()})))

    # Figure 1: pull-relative spray density by hand (both must peak on the positive side)
    fig, ax = plt.subplots(figsize=(7, 5))
    for st, c in (("L", C_L), ("R", C_R)):
        v = d24.filter(pl.col("stand") == st)["spray_pull"].to_numpy()
        ax.hist(v, bins=90, range=(-90, 90), density=True, histtype="step",
                color=c, lw=1.6, label=f"{st}HB (mean {v.mean():+.2f} deg)")
    ax.axvline(0, color="#8a8a8a", ls="--", lw=1)
    ax.set_xlabel("spray_pull (deg; POSITIVE = pulled)"); ax.set_ylabel("density")
    ax.set_title("2024 pull-relative spray - both hands must lean positive")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "spray_by_hand.png", dpi=120); plt.close(fig)

    # Figure 2: raw direction on home runs (the asymmetry BART needs `stand` to see)
    fig, ax = plt.subplots(figsize=(7, 5))
    hr24 = d24.filter(pl.col("events") == "home_run")
    for st, c in (("L", C_L), ("R", C_R)):
        v = hr24.filter(pl.col("stand") == st)["phi_raw"].to_numpy()
        ax.hist(v, bins=60, range=(-60, 60), density=True, histtype="step",
                color=c, lw=1.6, label=f"{st}HB HR (mean {v.mean():+.1f} deg)")
    ax.axvline(0, color="#8a8a8a", ls="--", lw=1)
    ax.set_xlabel("phi_raw (deg; negative = LEFT field)"); ax.set_ylabel("density")
    ax.set_title("2024 home runs, RAW direction - opposite peaks by hand")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "spray_hr_raw_direction.png", dpi=120); plt.close(fig)

    (outdir / "spray_qc.json").write_text(json.dumps(
        {"per_season": per_season, "named_2024": named,
         "n_switch_2024": len(switch_ids), "switch_by_stand": sw.to_dicts(),
         "gates": gates}, indent=2, default=float))
    failed = [g["name"] for g in gates if not g["pass"]]
    print(f"  wrote {outdir}/spray_qc.json and 2 figures")
    if failed:
        print(f"\n  HARD GATE FAILURES: {failed}")
        print("  DO NOT RUN THE STAGE-3 FIT. The mirror is wrong; check src/prep._spray_cols:")
        print("    - RHB pull to LEFT, so spray_pull = -phi_raw for stand == 'R'")
        print("    - phi_raw = atan2(hc_x - 125.42, 198.27 - hc_y): x-term FIRST")
        raise SystemExit(1)
    print("\n  Sign QC PASS. Safe to spend 27 minutes on the refit.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run the QC**

Run: `.venv/bin/python scripts/qc_spray.py`
Expected: S1–S6 all PASS, exit 0, seconds of runtime. The reported numbers should land on the
anchor table above (mean pull L ≈ +6.8…+7.5, R ≈ +3.2…+3.6; HR means +16…+20; `|φ|>45` ≈ 9%).

- [ ] **Step 5.3: Eyeball both figures**

Open `results/stage2_rebuild/figures/spray_by_hand.png` — both curves must lean right of zero.
Open `spray_hr_raw_direction.png` — the two HR curves must peak on *opposite* sides of zero. If they
peak on the same side, `stand` is not entering the mirror and S1 is passing by coincidence.

- [ ] **Step 5.4: Commit**

```bash
git add scripts/qc_spray.py results/stage2_rebuild/
git commit -m "feat(qc): spray sign gates S1-S6 + figures, run before the surface refit"
```

---

### Task 6: Variant-aware evaluation helpers (TDD)

**Files:**
- Modify: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

`contact_grids` hard-codes 3-column arrays and `variable_importance` hard-codes the 3 feature labels.
Both need the 5-feature variant. The grids also become the E7 sprint-migration diagnostic: the same
topped grounder, hit **pulled** vs **opposite**, across the sprint grid.

- [ ] **Step 6.1: Write the failing tests (append to `tests/test_evaluate.py`)**

```python
from src.evaluate import contact_grids


def test_contact_grids_v0_unchanged():
    s, g, b = contact_grids((23.0, 31.0, 5))
    assert s.shape == (5,) and g.shape == (5, 3) and b.shape == (5, 3)
    assert g[0].tolist() == [85.0, -10.0, 23.0]
    assert b[0].tolist() == [103.0, 28.0, 23.0]


def test_contact_grids_spray_variant_adds_pulled_and_oppo_grounders():
    s, grids = contact_grids((23.0, 31.0, 5), variant="spray")
    assert set(grids) == {"grounder_pull", "grounder_oppo", "barrel_pull"}
    for name, X in grids.items():
        assert X.shape == (5, 5)
        assert X[0, 3] == 1.0                      # stand_R: all grids are RHB
        assert X[0, 4] == 23.0                     # sprint speed is the last column
    assert grids["grounder_pull"][0].tolist() == [85.0, -10.0, 20.0, 1.0, 23.0]
    assert grids["grounder_oppo"][0].tolist() == [85.0, -10.0, -20.0, 1.0, 23.0]
    assert grids["barrel_pull"][0].tolist() == [103.0, 28.0, 20.0, 1.0, 23.0]


def test_la_spray_grid_is_row_major_over_la_then_spray():
    from src.evaluate import la_spray_grid

    la_ax, sp_ax, X = la_spray_grid(la=(0.0, 30.0, 4), spray=(-20.0, 20.0, 3))
    assert la_ax.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert sp_ax.tolist() == [-20.0, 0.0, 20.0]
    assert X.shape == (12, 5)
    # row-major: LA is the slow axis, spray the fast one -- the reshape in the figure
    # code is X[:, k].reshape(len(la_ax), len(sp_ax)) and depends on exactly this
    assert X[:3, 1].tolist() == [0.0, 0.0, 0.0]
    assert X[:3, 2].tolist() == [-20.0, 0.0, 20.0]
    assert X[3, 1] == 10.0
    assert (X[:, 0] == 103.0).all() and (X[:, 3] == 1.0).all()
```

Note `test_contact_grids_v0_unchanged` is a **regression guard, not a TDD red test** — it passes
before the change and must keep passing after, which is exactly its job: proving the v0 return
signature did not move.

- [ ] **Step 6.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: `test_contact_grids_spray_variant_adds_pulled_and_oppo_grounders` FAILs with
`TypeError: contact_grids() got an unexpected keyword argument 'variant'`, and
`test_la_spray_grid_is_row_major_over_la_then_spray` FAILs with
`ImportError: cannot import name 'la_spray_grid'` (its import is function-local, so it fails at call
time, not collection). `test_contact_grids_v0_unchanged` passes — it is the regression guard.

- [ ] **Step 6.3: Implement — replace `contact_grids` in `src/evaluate.py`**

```python
SPRAY_PULL_DEG = 20.0     # ~ the HR-region pull angle measured on 2022-25


def contact_grids(grid_cfg: tuple[float, float, int], variant: str = "v0"):
    """Fixed contact points crossed with a sprint-speed grid (spec §9.3):
    topped grounder (85, -10) and barrel (103, 28).

    variant 'v0'    -> (s, grounder(n,3), barrel(n,3)) -- unchanged, 3 features.
    variant 'spray' -> (s, {name: (n,5)}) with the SAME grounder hit PULLED (+20 deg)
    and OPPOSITE (-20 deg). All grids are RHB (stand_R = 1.0). The pull/oppo pair is
    the E7 diagnostic: once the surface can see direction, sprint speed's payoff should
    concentrate on pulled grounders (the ones a fast runner beats out)."""
    lo, hi, n = grid_cfg
    n = int(n)
    s = np.linspace(lo, hi, n)
    if variant == "v0":
        grounder = np.column_stack([np.full(n, 85.0), np.full(n, -10.0), s])
        barrel = np.column_stack([np.full(n, 103.0), np.full(n, 28.0), s])
        return s, grounder, barrel

    def g(ls: float, la: float, spray: float) -> np.ndarray:
        return np.column_stack([np.full(n, ls), np.full(n, la), np.full(n, spray),
                                np.ones(n), s])

    return s, {"grounder_pull": g(85.0, -10.0, SPRAY_PULL_DEG),
               "grounder_oppo": g(85.0, -10.0, -SPRAY_PULL_DEG),
               "barrel_pull": g(103.0, 28.0, SPRAY_PULL_DEG)}


def la_spray_grid(la: tuple[float, float, int] = (0.0, 45.0, 19),
                  spray: tuple[float, float, int] = (-45.0, 45.0, 37),
                  launch_speed: float = 103.0, sprint: float = 27.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LA x spray grid at fixed EV for the spec's HR-band partial-dependence plot
    (spec §"Recommended spec": "PDP/importance for spray (HR band in LA x spray)").
    Returns (la_axis, spray_axis, X (n_la*n_spray, 5)) in row-major order, RHB."""
    la_ax = np.linspace(*la[:2], int(la[2]))
    sp_ax = np.linspace(*spray[:2], int(spray[2]))
    L, S = np.meshgrid(la_ax, sp_ax, indexing="ij")
    n = L.size
    X = np.column_stack([np.full(n, launch_speed), L.ravel(), S.ravel(),
                         np.ones(n), np.full(n, sprint)])
    return la_ax, sp_ax, X
```

Then make the importance labels a parameter — change the signature and drop the hard-coded list:

```python
def variable_importance(figdir: Path, model, idata, X: np.ndarray,
                        labels: list[str] | None = None) -> dict:
```
and inside, replace `labels = ["launch_speed", "launch_angle", "sprint_speed"]` with
`labels = labels or ["launch_speed", "launch_angle", "sprint_speed"]`.

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_evaluate.py -v` — Expected: all pass
Run: `.venv/bin/pytest` — Expected: **61 passed**

- [ ] **Step 6.5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): variant-aware contact grids (pull vs oppo), LA x spray PDP grid, importance labels"
```

---

### Task 7: Wire the spray variant into the orchestrator + draw persistence

**Files:**
- Modify: `scripts/run_v0.py`, `.gitignore`

One orchestrator, two variants. This is what makes "same machinery, same train-subsample protocol and
holdout as v0" a fact rather than an intention.

- [ ] **Step 7.1: Add the gitignore rule FIRST** (before anything can write 390 MB into git)

Append to `.gitignore`:
```
results/**/*.npy
results/**/*.pkl
results/**/ev_draws_keys_*.parquet
```

The key parquets need the rule too: 363,595 + 122,006 rows across 8 columns is ~10-15 MB of derived
data, and Step 8.5 runs `git add results/stage_C_spray`. Every parquet currently tracked under
`results/` is ~110 KB, so committing these would be a 100x departure from repo practice — and
permanent.

Verify:
```bash
git check-ignore -v results/stage_C_spray/ev_draws_train.npy \
    results/stage_C_spray/all_trees.pkl \
    results/stage_C_spray/ev_draws_keys_train.parquet
```
Expected: prints a matching rule for each of the three.

- [ ] **Step 7.2: Edit `scripts/run_v0.py`**

1. **Argparse** — after the `--stage` argument:
```python
    ap.add_argument("--variant", choices=["v0", "spray"], default="v0",
                    help="v0 = 3 features (frozen anchor path); spray = 5 features")
    ap.add_argument("--persist-draws", type=int, default=200,
                    help="thinned per-event value draws to persist (spray variant only)")
```

2. **Stage dir + feature list** — replace the `stage_dir` assignment:
```python
    suffix = "" if args.variant == "v0" else f"_{args.variant}"
    stage_dir = cfg.results_dir / f"stage_{args.stage}{suffix}"
    features = prep.VARIANT_FEATURES[args.variant]
```
and add `"variant": args.variant, "features": features` to the `metrics` dict literal.

**`results/stage_C` is never overwritten** — the spray run writes `results/stage_C_spray`. That is
load-bearing: the ELPD anchor lives in the v0 directory.

3. **Spray columns in `prep_bbe`** — the imputation table must be built on TRAIN only and reused for
the holdout, so build it once between the two `prep_bbe` calls. Replace the `bbe_train`/`bbe_hold`
block:
```python
    bbe_train = prep_bbe(df_train, "train")
    bbe_hold = prep_bbe(df_hold, "holdout")
    if args.variant == "spray":
        # Imputation table from TRAINING seasons only, applied to both, so the holdout
        # never imputes itself. hc missingness is ~0.04% of BBE; rows are imputed and
        # flagged, NEVER dropped -- the holdout must stay at 122,006 events for the
        # -80107 ELPD anchor to be comparable.
        cell, hand = prep.spray_impute_table(bbe_train)
        bbe_train = prep.add_spray(bbe_train, cell, hand)
        bbe_hold = prep.add_spray(bbe_hold, cell, hand)
        metrics["hc_imputed_rate"] = {
            "train": float(bbe_train["hc_imputed"].mean()),
            "holdout": float(bbe_hold["hc_imputed"].mean()),
        }
        assert bbe_hold.height == 122006, f"holdout BBE moved: {bbe_hold.height}"
```

4. **Feature builds** — every `prep.build_features(...)` call takes `features`:
`prep.build_features(fit_df, features)`, `prep.build_features(pt_train, features)`,
`prep.build_features(pt_hold, features)`.

5. **Importance labels** — `evaluate.variable_importance(figdir, mdl, idata, X_fit, features)`.

6. **Localization grids** — replace the `contact_grids` block:
```python
    if args.variant == "v0":
        s_grid, X_g, X_b = evaluate.contact_grids(cfg.sprint_grid)
        b_g = model_mod.predict_and_reduce(mdl, idata, X_g, None, w, cfg, cfg.seed)
        b_b = model_mod.predict_and_reduce(mdl, idata, X_b, None, w, cfg, cfg.seed)
        grounder_ev, barrel_ev = b_g.ev_draws, b_b.ev_draws
    else:
        s_grid, grids = evaluate.contact_grids(cfg.sprint_grid, variant="spray")
        gd = {k: model_mod.predict_and_reduce(mdl, idata, X, None, w, cfg, cfg.seed).ev_draws
              for k, X in grids.items()}
        grounder_ev, barrel_ev = gd["grounder_pull"], gd["barrel_pull"]
        # E7: sprint speed's payoff should concentrate on PULLED grounders
        slopes = {k: float(np.polyfit(s_grid, v.mean(axis=0), 1)[0]) for k, v in gd.items()}
        metrics["sprint_migration"] = {
            "slopes_per_ftps": slopes,
            "pull_minus_oppo": slopes["grounder_pull"] - slopes["grounder_oppo"],
        }
        print("sprint migration:", metrics["sprint_migration"])
```
and pass `grounder_ev, barrel_ev` into `evaluate.localization(...)` in place of
`b_g.ev_draws, b_b.ev_draws`.

Immediately after that, add the spec's HR-band partial-dependence plot (spec §"Recommended spec":
*"PDP/importance for spray (HR band in LA × spray)"*). It needs the live model, so it cannot be
deferred to a later plan:
```python
    if args.variant == "spray":
        la_ax, sp_ax, X_pdp = evaluate.la_spray_grid()
        p_pdp = model_mod.predict_and_reduce(
            mdl, idata, X_pdp, None, w, cfg, cfg.seed).p_mean      # (n, K)
        hr = p_pdp[:, prep.CLASS_NAMES.index("home_run")].reshape(len(la_ax), len(sp_ax))
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(sp_ax, la_ax, hr, shading="nearest", cmap="viridis")
        fig.colorbar(im, ax=ax, label="P(home run)")
        ax.set_xlabel("spray_pull (deg; POSITIVE = pulled)")
        ax.set_ylabel("launch_angle (deg)")
        ax.set_title("HR band in LA x spray (RHB, EV 103 mph)")
        fig.tight_layout()
        fig.savefig(figdir / "pdp_la_spray_hr.png", dpi=120)
        plt.close(fig)
        metrics["pdp_hr_band"] = {
            "max_p_hr": float(hr.max()),
            "argmax_la": float(la_ax[hr.max(axis=1).argmax()]),
            "argmax_spray": float(sp_ax[hr.max(axis=0).argmax()]),
        }
        print("HR band peak:", metrics["pdp_hr_band"])
```
**New imports `run_v0.py` needs** (it currently has none of these): `import hashlib` (the
order digest in item 7), `import pickle` (item 11), and `import matplotlib` /
`matplotlib.use("Agg")` / `import matplotlib.pyplot as plt` for this figure, mirroring
`src/evaluate.py:7-10`.

**Expected shape of the result:** the HR probability should peak in a band around LA 25–30° and
lean toward *pulled* spray (positive), matching the measured HR anchors (+16…+20° mean pull). A
heatmap that is flat in spray means the feature is not reaching the model.

7. **Draw persistence (E6).**

**Read the shared insertion rule in item 9 before placing this block.** Persistence writes 389 MB
and marginalization costs 30 minutes; both must run **after** the ELPD verdict is on disk, not
before it. Neither is needed by anything downstream in `run_v0.py`.

The block itself:
```python
    if args.variant == "spray":
        # Stage-4 prerequisite: per-event value draws. Stage 4 folds their between-draw
        # variance into S_i[0,0], which is what finally makes interval coverage testable.
        # Persisting these is the whole reason the refit is worth doing once, not twice.
        def persist(tag: str, ev: np.ndarray, pt: pl.DataFrame) -> dict:
            k = min(args.persist_draws, ev.shape[0])
            idx = np.linspace(0, ev.shape[0] - 1, k).astype(int)
            arr = np.ascontiguousarray(ev[idx], dtype=np.float32)
            # Assert BEFORE writing 291 MB, not after.
            assert arr.shape[1] == pt.height, "draw columns must align with key rows"
            np.save(stage_dir / f"ev_draws_{tag}.npy", arr)
            pt.select("batter", season=pl.col("game_year"),
                      woba_denom=pl.col("woba_denom"),
                      launch_speed=pl.col("launch_speed"),
                      launch_angle=pl.col("launch_angle"),
                      spray_pull=pl.col("spray_pull"), stand_R=pl.col("stand_R"),
                      sprint_speed=pl.col("sprint_speed"),
                      hc_imputed=pl.col("hc_imputed")
                      ).with_row_index("row").write_parquet(
                          stage_dir / f"ev_draws_keys_{tag}.parquet")
            # The assert above is near-tautological (both derive from the same frame);
            # what actually matters is row ORDER. Stamp a checkable key so Stage 4 can
            # detect a reordering rather than trusting the contract.
            return {"shape": list(arr.shape), "mb": round(arr.nbytes / 1e6, 1),
                    "draw_index": idx.tolist(),
                    "batter_order_digest": hashlib.sha256(
                        pt["batter"].to_numpy().tobytes()).hexdigest()[:16]}

        metrics["persisted_draws"] = {
            "train": persist("train", b_train.ev_draws, pt_train),
            "holdout": persist("holdout", b_hold.ev_draws, pt_hold),
        }
        np.save(stage_dir / "lppd_i_holdout.npy", b_hold.lppd_i.astype(np.float64))
        print("persisted draws:", metrics["persisted_draws"])
```

The `.npy` axis-1 ordering is the row order of `pt_train`/`pt_hold`, which is exactly the order the
key parquet records — the assert and the `row` index make that contract explicit for Stage 4.

8. **RESULTS.md tag** — `update_results_md(cfg.results_dir, f"{args.stage}{suffix}", [...])` so the
spray run gets its own `<!-- stage_C_spray -->` block instead of overwriting v0's.

9. **Spray marginalization (the A/B's B side)** — add the flag next to `--persist-draws`:
```python
    ap.add_argument("--marginalize-spray", type=int, default=0, metavar="M",
                    help="spray-marginalized per-event values via M equal-mass league "
                         "quantiles per (EV, LA, stand) cell; 0 = skip. Costs M x the "
                         "prediction pass (~30 min at M=9 on Stage C)")
```
and insert this block directly after the draw-persistence block from item 7, at the shared insertion
point defined below, **while `mdl` is still live** — the fitted trees hang off the model object and
cannot be recovered from `idata.nc` later:
```python
    if args.variant == "spray" and args.marginalize_spray:
        # Design risk 2: conditioning a player rollup on per-ball direction credits spray
        # LUCK. The marginalized value replaces v(x_e) with its league average over spray
        # given EV x LA x stand -- no refit, just M extra prediction passes.
        M = args.marginalize_spray
        qs = np.linspace(0.5 / M, 1 - 0.5 / M, M)          # M equal-mass quantiles
        si = features.index("spray_pull")
        src = bbe_train.with_columns(
            _ev=(pl.col("launch_speed") // prep.SPRAY_EV_BIN).cast(pl.Int32),
            _la=(pl.col("launch_angle") // prep.SPRAY_LA_BIN).cast(pl.Int32))
        qt = (src.group_by("_ev", "_la", "stand_R")
                 .agg([pl.col("spray_pull").quantile(q).alias(f"q{i}")
                       for i, q in enumerate(qs)], _n=pl.len())
                 .filter(pl.col("_n") >= prep.SPRAY_MIN_CELL).drop("_n")
                 .sort("_ev", "_la", "stand_R"))     # polars group_by is not order-stable
        marg, sparse_rate = {}, {}
        for tag, pt in (("train", pt_train), ("holdout", pt_hold)):
            j = (pt.with_row_index("_r").with_columns(
                     _ev=(pl.col("launch_speed") // prep.SPRAY_EV_BIN).cast(pl.Int32),
                     _la=(pl.col("launch_angle") // prep.SPRAY_LA_BIN).cast(pl.Int32))
                   .join(qt, on=["_ev", "_la", "stand_R"], how="left").sort("_r"))
            assert j.height == pt.height, "marginalization join changed row count"
            Q = j.select([f"q{i}" for i in range(M)]).to_numpy()          # (n, M)
            base = j.select(features).to_numpy().astype(np.float64)       # (n, 5)
            # Capture sparsity BEFORE the fill -- after np.where, Q is all-finite and the
            # rate would be identically 0.0, silently killing the no-op diagnostic that
            # Steps 7.4 and 8.1 depend on.
            sparse_rate[tag] = float(np.mean(~np.isfinite(Q)))
            # cells too sparse for quantiles keep their OBSERVED spray (identity)
            Q = np.where(np.isfinite(Q), Q, base[:, si:si + 1])
            acc = np.zeros(base.shape[0])
            for m in range(M):
                Xm = base.copy()
                Xm[:, si] = Q[:, m]
                acc += model_mod.predict_and_reduce(
                    mdl, idata, Xm, None, w, cfg, cfg.seed).ev_draws.mean(axis=0)
            marg[tag] = acc / M
            np.save(stage_dir / f"ev_marginalized_{tag}.npy", marg[tag].astype(np.float32))
        metrics["marginalize_spray"] = {
            "M": M,
            "sparse_cell_rate": sparse_rate,
            "mean_abs_shift_train": float(np.abs(
                marg["train"] - b_train.ev_draws.mean(axis=0)).mean()),
        }
        print("marginalized:", metrics["marginalize_spray"])
```
The `ev_marginalized_{tag}.npy` arrays are row-aligned with the *same* `ev_draws_keys_{tag}.parquet`
written in item 7 — one alignment contract for both products.

**SHARED INSERTION POINT for items 7 and 9 (this matters more than it looks).** Put **both** blocks
— draw persistence first, then marginalization — **after** `metrics["undercorrection_gb_holdout"] = ...`
(`run_v0.py:194`), *not* after the rollup at `run_v0.py:161`.

`run_v0.py:163-194` is the whole evaluation block — replication, calibration, **ELPD**, localization,
importance — and `metrics.json` is only written at `run_v0.py:199`. Placing either block earlier
means a failure in optional work discards the E1 verdict from a 27-minute fit. Both failure modes are
live: marginalization is a 30-minute pass that can crash, and persistence writes 389 MB into a run
that already needs ~4 GB for `idata.nc` — Step 10.4 warns about disk, and running out of it during
persistence is precisely how the ELPD result would be lost. Nothing between `:161` and `:199` reads
either output.

For the same reason, add a durability write immediately after
`metrics["elpd"] = evaluate.elpd_metrics(...)` (`run_v0.py:177`):
```python
    # Durability: the thesis metric is on disk before any optional work runs.
    (stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
```
(The final write at `run_v0.py:199` supersedes it.)

10. **Free ~1 GB before marginalizing.** At that point `b_train.ev_draws` (500 × 363,595 f32 = 727 MB),
`b_hold.ev_draws` (244 MB) and `ev_all` (970 MB, `run_v0.py:159`) are all live, and each
marginalization iteration allocates another 727 MB. `ev_all` is dead after
`table.write_parquet(...)` — add `del ev_all` there.

11. **Make the fit recoverable.** Right after `model_mod.save_idata(idata, stage_dir / "idata.nc")`
(`run_v0.py:116`), pickle the fitted trees:
```python
    if args.variant == "spray":
        # save_idata writes only the InferenceData; the fitted trees live on the live op
        # (src/model.py:106) and are otherwise unrecoverable, which is why the fit and the
        # marginalization must share one process. Pickling them demotes a failed 60-minute
        # run to a 30-minute one. Best-effort ONLY: this sits immediately after the most
        # expensive step in the plan, so an unpicklable tree object must not kill the run.
        try:
            with open(stage_dir / "all_trees.pkl", "wb") as f:
                pickle.dump(mdl["mu"].owner.op.all_trees, f, protocol=4)
        except Exception as exc:
            print(f"WARN: could not pickle all_trees ({type(exc).__name__}: {exc}); "
                  f"a re-run would need a full refit")
```
**How to actually use it if the run dies later:** `predict_and_reduce` needs a live `model`, so
recovery is not a drop-in. Load the pickle and call the tree sampler directly, mirroring
`src/model.py:100-110` — `pymc_bart.utils._sample_posterior(all_trees, X=X_new, rng=rng, size=S,
shape=K)`, which returns `(S, n, K)` and must be transposed to `(S, K, n)`. Note that the Step 7.4
Stage-A smoke exercises this write path, so confirm `results/stage_A_spray/all_trees.pkl` exists
there before trusting it at Stage C.

- [ ] **Step 7.3: Prove the v0 path is unchanged, cheaply**

Run: `.venv/bin/python scripts/run_v0.py --stage A`
Expected: completes in ~1 min, writes `results/stage_A/`, and `git diff --stat results/stage_A` shows
only run-to-run noise (runtimes) — the feature list, fit rows (5,000) and class distribution must be
identical. **If Stage A's numbers moved, the refactor broke the v0 path — stop.**

- [ ] **Step 7.4: Smoke the spray path at Stage A — including marginalization**

Run: `.venv/bin/python scripts/run_v0.py --stage A --variant spray --marginalize-spray 3`
Expected: ~2–4 min. Confirm in `results/stage_A_spray/metrics.json`:
`features` has 5 entries; `hc_imputed_rate` ≈ 0.0004; `sprint_migration` is present;
`persisted_draws.holdout.shape` is `[200, 20000]` (Stage A caps prediction at 20,000);
`marginalize_spray.mean_abs_shift_train` is > 0 and < 0.5 (a shift of exactly 0 means the
marginalization is a no-op — most likely `si` is indexing the wrong column, or every cell fell to the
sparse-identity branch, which `sparse_cell_rate` will show); `oos_verification.pass` is `true`;
`pdp_hr_band` is present. Also confirm on disk: both `ev_draws_*.npy`,
`ev_marginalized_holdout.npy` with length 20,000, `all_trees.pkl` (the best-effort write actually
succeeded), and `figures/pdp_la_spray_hr.png`.

**This smoke is not optional.** It exercises every line of the ~60-minute Stage-3 wiring in ~3
minutes, so a shape, column-index or join bug surfaces before the expensive run.

- [ ] **Step 7.5: Commit**

```bash
git add .gitignore scripts/run_v0.py results/stage_A results/stage_A_spray results/RESULTS.md
git commit -m "feat(run_v0): --variant spray (5 features), pull/oppo grids, draw persistence, spray marginalization"
```

---

### Task 8: The Stage-C spray refit — the ELPD gate (E1–E7)

**Files:**
- Modify: `results/RESULTS.md` (written by the runner)

**~60 minutes, one invocation.** Tasks 1–7 must be green first: `.venv/bin/pytest` at 61 passed,
R1–R6 PASS, S1–S6 PASS, the Stage-A spray smoke clean.

- [ ] **Step 8.1: Run the fit + marginalization**

Run:
```bash
.venv/bin/python scripts/run_v0.py --stage C --variant spray --acknowledge-runtime \
    --marginalize-spray 9
```
Expected: ~27 min fit + ~30 min marginalization ≈ 60 min total. `--acknowledge-runtime` is passed
defensively — v0's Stage-C estimate was 28.46 min, just under the bar, so the guard at
`run_v0.py:109` may not fire at all; the flag costs nothing and prevents a stall. **Do not split this
into two commands** — the marginalization needs the live `model` object and would otherwise force a refit.

Expect on the way past:
- `WARN (BART mu R-hat is structural, not a convergence stop)` — **normal, not a failure.** mu-cell
  R-hat is not identified at the cell level. E4 (`verify_oos_mechanism`, corr ~1.0) is the real gate.
- `oos_verification` must print `pass: true`. If it does not, stop — the stored-trees predictor is
  the only working OOS path in pymc-bart 0.12 (`pm.set_data` silently freezes `mu` and returns the
  in-sample trace regardless of `X_new`, a wrong answer rather than an exception).
- `persisted draws` printing `[200, 363595]` (~291 MB) and `[200, 122006]` (~98 MB).
- `marginalized:` printing a non-zero `mean_abs_shift_train` and a small `sparse_cell_rate`. A shift
  of exactly 0 means the marginalization silently no-oped — investigate before trusting Task 9.

- [ ] **Step 8.2: Score the gates**

Run:
```bash
.venv/bin/python - <<'PY'
import json
m = json.load(open("results/stage_C_spray/metrics.json"))
v = json.load(open("results/stage_C/metrics.json"))
e, e0 = m["elpd"], v["elpd"]
d = e["elpd_lppd"] - e0["elpd_lppd"]
print(f"E1 ELPD {e['elpd_lppd']:.1f} vs anchor {e0['elpd_lppd']:.1f} -> delta {d:+.1f} nats "
      f"({d/e['n_events']:+.5f}/event); n {e['n_events']} (want 122006)")
print("E1", "PASS" if d >= 1000 and e["n_events"] == e0["n_events"] else "FAIL")
print(f"E2 ECE {m['calibration']['ece_weighted']:.6f} vs {v['calibration']['ece_weighted']:.6f} "
      f"(cap 0.046456)", "PASS" if m['calibration']['ece_weighted'] <= 0.0464559 else "FAIL")
for c in ("triple", "home_run", "double"):
    a = m["calibration"]["per_class"][c]["brier"]; b = v["calibration"]["per_class"][c]["brier"]
    print(f"E3 brier[{c}] {a:.6f} vs {b:.6f} ({a-b:+.6f})")
print("E4 oos", m["oos_verification"])
print("E5 sanity", m["sanity_warnings"])
print("E6 draws", m["persisted_draws"]["train"]["shape"], m["persisted_draws"]["holdout"]["shape"])
print("E7 importance", m["variable_importance"].get("raw", {}).get("indices"),
      "features", m["features"])
print("E7 sprint migration", m["sprint_migration"])
print("E7 HR-band PDP", m["pdp_hr_band"])
print("E8 marginalization", m["marginalize_spray"])
PY
```

Anchors this compares against: ELPD **−80107.495** over **122,006** events; ECE **0.042233**;
Brier triple **0.005013**, HR **0.025344**, double **0.052073**; v0 grounder sprint slope
**0.0023488**.

- [ ] **Step 8.3: If E1 FAILS — stop and report**

Write the numbers, then work the checklist in the gate table above **in order**. Do not raise
`m_trees`, `draws`, or the subsample: that breaks the protocol match and the anchor comparison with
it. A documented non-beat is a valid outcome of this plan; a rescued one is not.

- [ ] **Step 8.4: Eyeball the figures**

`results/stage_C_spray/figures/`: `calibration_reliability.png` (5 panels, none collapsed),
`replication_event_holdout.png` (r should stay ≈0.91 — the model is no longer *trying* to match
Savant's spray-blind number, so a small drop here alongside an ELPD gain is expected and correct, not
a regression), `sprint_localization_curves.png`, `variable_importance.png`, and `pdp_la_spray_hr.png` (the HR band
should sit around LA 25-30° and lean toward positive spray; flat in spray means the feature is not
reaching the model).

- [ ] **Step 8.5: Commit** (the `.npy` files are gitignored; `idata.nc` already is)

```bash
git add results/stage_C_spray results/RESULTS.md
git commit -m "feat(surface): Stage-C spray refit — 5-feature BART, ELPD vs the -80107 anchor"
```

---

### Task 9: Spray-conditioned vs spray-marginalized rollup A/B (E8)

**Files:**
- Create: `scripts/rollup_ab.py`

The design's risk 2: conditioning a player rollup on per-ball direction credits spray *luck*. Public
spray-adjusted xwOBA variants typically describe the same season better without predicting the next
one better. **Do not assume the conditioned rollup wins.**

Note the scope boundary: Level 2 currently consumes *public* Savant per-event xwOBA, not the model
rollup, so pushing the winner through `talent2` is Stage 4's job. What Stage 3 can and should settle
is which rollup predicts next-season actual wOBA better — raced directly, no talent layer needed.

Task 7 already produced `ev_marginalized_{tag}.npy` during the Task-8 run. **This task loads saved
arrays and needs no model, no `idata`, and no refit** — it runs in seconds and is safe to iterate on.

- [ ] **Step 9.1: Implement `scripts/rollup_ab.py`**

Structure (~180 lines; build it in this order):

1. **Load, per `tag` in (`train`, `holdout`):** `results/stage_C_spray/ev_draws_keys_{tag}.parquet`
   (carries `batter`, `season`, `woba_denom`, `row`), `ev_draws_{tag}.npy` (200, n — the
   spray-conditioned draws) and `ev_marginalized_{tag}.npy` (n,). Assert
   `arr.shape[-1] == keys.height` for both before doing anything else — the alignment contract is
   positional and an off-by-one here is silent.

2. **Two rollups** via the existing `rollup.player_rollup(ev_draws, bbe_keys, non_bbe)`:
   - conditioned: pass the (200, n) draws straight through;
   - marginalized: pass `marg[None, :]` (a 1-draw stack) — `player_rollup` handles `S == 1` (it
     zeroes `xwoba_sd`), and only the mean is needed for the race.

   `non_bbe` comes from `prep.build_non_bbe_pa` over the concatenated pitch frames for
   `cfg.all_seasons`, exactly as `run_v0.py:98` builds it. Concatenate the train and holdout key
   frames first so each player-season is rolled up once — and concatenate the arrays **in the same
   order**: `keys = pl.concat([k_train, k_hold]).drop("row")`,
   `cond = np.concatenate([d_train, d_hold], axis=1)`, `marg = np.concatenate([m_train, m_hold])`.
   Assert `cond.shape[1] == marg.size == keys.height` after concatenating, not just before.
   Drop `row` as shown: each parquet restarts its index at 0, so the concatenated column would carry
   duplicates and quietly undercut the positional-alignment contract. (`player_rollup` never reads
   it — it only needs `batter`, `season`, `woba_denom`.)

3. **Race** `{xwoba_conditioned, xwoba_marginalized, xwoba_v0, xwoba_savant}` against next-season
   actual wOBA. `xwoba_v0` is `xwoba_mean` from the frozen `results/stage_C/player_table.parquet`;
   `xwoba_savant` is that table's `xwoba_savant`.

   **The exact signatures, because they do not compose without an adapter:**
   - Imports need the script dir on the path first, as every other runner does:
     `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` **and**
     `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))`, then
     `from benchmark_vs_savant import actual_woba, _pearson, _calibrated_rmse` and
     `from run_talent2 import make_pairs`.
   - `actual_woba(raw_dir: Path, seasons: list[int]) -> (batter, season, **pa**, actual_woba)` —
     lowercase `pa`.
   - `make_pairs(base, seasons_t, pa_t_floor, min_pa_next)` reads **`pl.col("PA")`** off `base` and
     emits `target` / `pa_next`. **So you must rename `pa` → `PA` on the joined frame**, or the join
     silently produces an empty pair set.
   - `base` must be one row per `(batter, season)` carrying exactly: `batter`, `season`, `PA`,
     `actual_woba`, and the four predictor columns. **Select and rename before joining** — all three
     frames collide on `PA`/`xwoba_mean` (both rollups emit `PA`, `xwoba_mean`, `xwoba_sd`,
     `xwoba_q05`, `xwoba_q95`; the frozen v0 table emits `PA`, `xwoba_mean`, `xwoba_savant`), and
     polars will silently suffix the duplicates with `_right`:
     ```python
     cond_t = roll_cond.select("batter", "season", "PA", xwoba_conditioned="xwoba_mean")
     marg_t = roll_marg.select("batter", "season", xwoba_marginalized="xwoba_mean")
     v0_t   = (pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet")
                 .select("batter", "season", "xwoba_savant", xwoba_v0="xwoba_mean"))
     act_t  = actual_woba(cfg.raw_dir, cfg.all_seasons).select(
                  "batter", "season", "actual_woba")          # drop lowercase `pa`
     base = (cond_t.join(marg_t, on=["batter", "season"], how="inner")
                   .join(v0_t,   on=["batter", "season"], how="inner")
                   .join(act_t,  on=["batter", "season"], how="inner")
                   .sort("batter", "season"))
     ```
     `PA` comes from the rollup (capital, as `make_pairs` expects); dropping `actual_woba`'s
     lowercase `pa` sidesteps the rename entirely.
   - `_pearson(x, y)` and `_calibrated_rmse(pred, target)` take plain numpy arrays.

4. **Report:** pooled PA≥100 and PA≥30, by-band (30–60, 60–100, 100–250, 250+), all pairs
   (22→23, 23→24, 24→25). **RMSE primary, r secondary** — r is affine-invariant and cannot see
   better shrinkage within a band; Phase 1 already taught that lesson. Add a paired bootstrap
   (5,000 reps, seed 42) of conditioned − marginalized on the PA≥30 pool, reporting mean, 95% CI and
   `frac_better`.

5. **Also report the descriptive side:** same-season correlation of each rollup with Savant's public
   xwOBA at PA≥100. The design predicts an inversion — conditioned describing better while
   marginalized predicts better. **Both are legitimate products; label them, do not pick a "winner"
   overall.** Only the *predictive* winner designates Stage 4's talent input.

6. **Figure** `results/rollup_ab/figures/next_season_rmse_by_band.png`: grouped bars, calibrated RMSE
   by PA band, four predictors, colors from `benchmark_vs_savant` (`C_MODEL`, `C_SAVANT`, `C_NAIVE`).

7. **Write** `results/rollup_ab/rollup_ab_metrics.json` and `marginalized_values.parquet`
   (`batter`, `season`, `PA`, both rollups) — Stage 4 consumes the latter.

- [ ] **Step 9.2: Run the A/B and record the verdict**

Run: `.venv/bin/python scripts/rollup_ab.py`
Expected: seconds; writes `results/rollup_ab/`. Record which rollup wins next-season calibrated RMSE
at PA≥30 and PA≥100 — that is Stage 4's designated talent input. State the paired-bootstrap CI
alongside: with n≈1,000 pairs a difference under ~0.001 RMSE is not resolvable and must be reported
as a tie, not a win.

- [ ] **Step 9.3: Commit**

```bash
git add scripts/rollup_ab.py results/rollup_ab
git commit -m "feat(rollup): spray-conditioned vs spray-marginalized A/B on next-season wOBA"
```

---

### Task 10: Documentation + final verification

**Files:**
- Create: `results/stage2_rebuild/NOTES.md`, `results/rollup_ab/NOTES.md`
- Modify: `results/RESULTS.md`

- [ ] **Step 10.1: Write `results/stage2_rebuild/NOTES.md`** — mirror the voice of
`results/talent/NOTES.md`. Cover: why the rebuild needed a reproduction gate at all (the upstream
cache can move under you; adding columns cannot); the R1–R6 outcomes with numbers; the spray
transform with its sign convention spelled out in words ("a right-handed batter pulls to left field,
so the mirror negates φ_raw for R"); the S1–S6 table with the per-season measurements; the hc
missingness figures and the impute-don't-drop decision **with the ELPD-anchor reason**; the 65
switch-hitter batter-seasons as the evidence that `stand` is per-event; and the ~9% of BBE with
|φ_raw| > 45° explained (caught fouls) so a future reader does not "fix" it.

- [ ] **Step 10.2: Write `results/rollup_ab/NOTES.md`** — the descriptive/predictive inversion in
plain language, the two rollups' next-season RMSE and r by band, the paired-bootstrap CI, the
same-season Savant correlations, and an explicit one-line verdict naming Stage 4's talent input. If
the conditioned rollup wins prediction, say so and note that it contradicts the design's expectation.

- [ ] **Step 10.3: Add a `results/RESULTS.md` section** "Phase 2 Stage 3 — spray surface": the
5-feature list, the ELPD delta vs −80107 with per-event nats, ECE and the triple Brier, variable
importance ordering, the sprint-migration numbers, the HR-band PDP, where the persisted draws live
and their exact alignment contract (axis 1 of `ev_draws_{tag}.npy` ↔ row order of
`ev_draws_keys_{tag}.parquet`), and the rollup A/B verdict.

**Record the surface-uncertainty caveat next to the draws** (spec Risk 3, and it must travel with the
`.npy` files because Stage 4 is a separate plan and a separate session): between-draw variance is
exactly right for *per-player* intervals, but surface errors are **correlated across players in the
same feature region** — these intervals must not be reused for league-aggregate claims without the
per-draw refit variant.

Also note that under `--variant spray`, `metrics["localization"]["grounder_slope_per_ftps"]` is the
**pulled**-grounder slope and is therefore *not* like-for-like with v0's spray-blind 0.0023488.

Add to the **"Deviations from spec/plan"** list:
- `FEATURES` split into `FEATURES_V0`/`FEATURES_SPRAY`; `build_features` takes the list, so one
  orchestrator serves both variants and protocol parity is structural rather than aspirational.
- `stand` enters BART as a numeric `stand_R` (1.0/0.0) column, not the raw string.
- The design's hc missing-flag is an **audit column, not a BART feature** (0.034–0.043% missingness;
  a flag feature would be split noise and would make it a 6-feature model).
- Level 2 was re-run in Stage 2, not Stage 3, because `run_talent2.py` consumes public Savant
  per-event xwOBA and never touches the surface.
- **The rollup A/B races the two rollups directly against next-season wOBA rather than feeding both
  through Level 2** as the spec's §"Rollup choice under spray" specifies. Reason: Level 2 currently
  consumes public Savant per-event xwOBA, so wiring the model rollup into it is itself Stage-4 work;
  the direct race answers the same question ("let next-season RMSE pick the talent input") without
  it. Stage 4 should confirm the choice survives the talent layer.
- Spray imputation uses the conditional **median**, not the spec's conditional **mean**, because ~9%
  of BBE sit outside the foul lines in hc coordinates (|φ_raw| > 45°, caught fouls) and the mean is
  sensitive to that tail.

- [ ] **Step 10.4: Final verification**

Run: `.venv/bin/pytest` — Expected: **61 passed**
Run: `.venv/bin/python scripts/qc_spray.py` — Expected: S1–S6 PASS (idempotent)
Run: `git status --porcelain` — Expected: nothing but intended files; **no `.npy`, `.pkl`, or `ev_draws_keys_*.parquet` staged**
Run: `du -sh results/stage_C_spray` — Expected: **≈4.1G as `du` prints it** (GiB), i.e. ~4.4 GB
decimal, plus the trees pickle whose size is unmeasured. `idata.nc` alone is 4,005,679,176 bytes —
`mu` is (K=5, n_fit=100000) over 2 chains × 500 draws regardless of feature count, which is why
`results/stage_C` also measures 3.7G — plus 291 MB + 98 MB of draws and ~2 MB marginalized. All
gitignored. **Confirm ~5 GB free before Task 8.**

- [ ] **Step 10.5: Commit**

```bash
git add results/stage2_rebuild/NOTES.md results/rollup_ab/NOTES.md results/RESULTS.md
git commit -m "docs: Stage 2 rebuild + sign QC and Stage 3 spray surface results"
```

---

## Stage 4 (context only — NOT in this plan)

Fold the persisted per-event draw variance into Level 2's `S_i[0,0]`; push the A/B-winning rollup
through `talent2` (which requires teaching `build_pa_measurements` to take model values instead of
`estimated_woba_using_speedangle`); validate 50/80/90% interval coverage by PA band within ±5pp —
the genuinely new Phase-2 deliverable, and the thing Phase 1 structurally cannot produce. Multi-season
pooling + age at Level 2 is the flagged follow-on after that.
