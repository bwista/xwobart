# xwobart Phase 2 / Stage 1 — Level-2 Joint-MVN Talent Model Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Phase 1's league-mean empirical-Bayes talent estimate to a **joint MVN measurement model** over (raw xwOBA, average exit velocity, barrel rate) per batter-season — so a low-PA hitter is shrunk toward what their fast-stabilizing peripherals imply instead of toward the flat league mean — validated against Phase 1's frozen numbers and against next-season actual wOBA.

**Architecture:** Per the accepted design (`docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md` §3), the three observed stats are jointly noisy measurements of correlated latent talents: `z_i ~ N((θ_i, ξ_i), S_i)` with per-player measurement covariance `S_i` estimated by a within-season bootstrap over the player's PAs, and `(θ_i, ξ_i) ~ N(μ_season, Σ_talent)` with per-season means and a shared talent covariance fit by marginal MLE. The posterior `E[θ|z]` leans on peripherals exactly when the xwOBA sample is small. **Crucially, `S_i`'s off-diagonals carry the shared sampling noise** (all three stats come from the same balls) — the design's #1 named risk is that ignoring them manufactures fake low-PA gains, so they are estimated explicitly and a zeroed-off-diagonal diagnostic is part of validation. Everything is closed-form Gaussian once hyperparameters are fit — no MCMC, no BART re-fit, fits in minutes. This is Stage 1 of Phase 2; Stages 2–4 (cache rebuild with hc_x/hc_y/stand, one 5-feature BART surface refit persisting per-event value draws, ELPD vs the −80107 anchor, surface-variance + spray rollup A/B + coverage) are **out of scope here** and get their own plan after this ships.

**Tech Stack:** Python 3.12, Polars (pandas only at boundaries), NumPy, SciPy (`scipy.optimize.minimize` — already installed transitively; made a direct requirement), matplotlib, pytest. Reuses `src.config`, `src.talent`, the slim Statcast caches in `data/raw/` (they already carry `launch_speed` and `launch_speed_angle`; barrel = `launch_speed_angle == 6`, verified league rate 7.8% on 2024), `results/talent/talent_table.parquet` (Phase-1 baseline), `results/stage_C/player_table.parquet` (names + Savant), and `scripts/benchmark_vs_savant.py` helpers (`actual_woba`, `_pearson`, `_calibrated_rmse`).

**Background (what was established before this plan):**
- Phase 1 (`src/talent.py`) shrinks raw xwOBA toward the season league mean. Frozen validation (in `results/talent/talent_metrics.json`): predicting next-season actual wOBA, pooled **PA_T≥100 (n=1060): talent r=0.4886, calibrated RMSE=0.034512** (raw 0.4835, Savant 0.4908); pooled **PA_T≥30 (n=1173): talent r=0.4669** (raw 0.4454, Savant 0.4521). 2,636 player-seasons; per-season μ≈0.305–0.318, τ≈0.031.
- The Phase-2 brainstorm (spec prompt + response in `docs/superpowers/specs/`) killed the old roadmap's "shrink toward the model's own xwOBA" variant (a structural no-op) and the event-level batter-intercept variant (models a channel with year-to-year r≈0.12 — noise). At PA 30–100, peripherals out-predict raw xwOBA next season (barrel rate r=0.244, avg EV r=0.227 vs raw xwOBA r=0.179), so they belong in the **prior**, entering at the player-season level.
- Design risk #1 (shared sampling noise) drives this plan's architecture: a naive "regress xwOBA on peripherals for a prior mean" fits correlated noise (β inflates, τ deflates, fictitious low-PA gains). The joint MVN with explicit off-diagonal `S_i` is immune by construction; the validation task includes the "zero the off-diagonals" tripwire.
- Phase-1 limitation #3 (`results/talent/NOTES.md`): degenerate se²→0 at tiny identical-value samples. Fixed here with a variance floor.

**Worktree note:** Per the repo convention (see `docs/superpowers/handoffs/2026-07-18-v1-handoff.md` and the Phase-1 plan), work happens directly on `main` — no worktree. Commit per task; push only when asked.

**Skills:** @superpowers-extended-cc:test-driven-development for every TDD task; @superpowers-extended-cc:verification-before-completion before claiming any task done; @superpowers-extended-cc:systematic-debugging when anything fails.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/talent2.py` | Pure logic: per-PA measurement frame (value + EV + barrel), per-player measurement triple + bootstrap covariance `S_i`, marginal-MLE hyperparameter fit, closed-form conditional posterior, table assembly with 1-D fallback |
| `tests/test_talent2.py` | Unit tests for every pure function (synthetic data, known answers, parameter recovery) |
| `scripts/run_talent2.py` | Orchestrator: build the Level-2 table from the slim caches, **L2a regression gates vs Phase 1**, **L2b validation races** (talent2 vs Phase-1 talent vs raw vs Savant), paired bootstrap deltas, ablations, off-diagonal diagnostic, figures, `results/talent2/` |
| `requirements.txt` | Add `scipy>=1.10` (direct dependency now) |
| `results/talent2/` (generated) | `talent2_table.parquet`, `l2a_table.parquet`, `talent2_metrics.json`, `figures/*.png`, `NOTES.md` |
| `results/RESULTS.md` | Add a "Level-2 talent model (joint MVN)" section |
| `docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md` | Supersession note pointing at the new design docs + this plan |

Data flow: slim caches → `build_pa_measurements` → (`player_measurements` + `bootstrap_S` → `assemble_measurements`) → `mvn_mle` (per-season μ, shared Σ, fit on PA≥100) → `mvn_posterior` → `build_talent2_table` → `run_talent2.py` (gates + validation + figures + docs).

**Naming/conventions:** mirror `src/talent.py` / `scripts/run_talent.py` exactly — pure functions in `src/`, orchestration + figures in `scripts/`, `Z90` for the 90% interval, figures reuse the run_talent color constants, tests are plain pytest functions on synthetic polars frames. Commands run from repo root with `.venv/bin/...`.

## Success gates (defined up front, from the design's §4)

| Gate | Criterion | Type |
|---|---|---|
| G1 (L2a regression) | 1-D variant reproduces Phase 1: table height 2,636; corr(θ_L2a, θ_Phase1) ≥ 0.999; validation r within ±0.005 of 0.4886 (PA≥100) and 0.4669 (PA≥30) | HARD |
| G2 (SE agreement) | Bootstrap xwOBA SE vs Phase-1 analytic SE: corr ≥ 0.98, median ratio ∈ [0.9, 1.1] (rows PA≥30) | HARD |
| G3 (low-PA win) | Pooled PA_T≥30, all pairs: talent2 beats Phase-1 talent on **both** r and calibrated RMSE (point estimates) | HARD |
| G4 (high-PA non-inferiority) | Pooled PA_T≥100: talent2 r ≥ Phase-1 talent r − 0.005 | HARD |
| G5 (noise tripwire) | Zeroed-off-diagonal refit must NOT show a larger PA≥30 r gain than the proper fit by > 0.005 (if it does, the gain is the shared-noise artifact — flag, investigate, do not ship the claim) | DIAGNOSTIC |
| G6 (honest split) | Model choices (ablations) scored on 22→23 + 23→24 only; 24→25 reported as confirmation; disagreement reported in NOTES, not silently dropped | PROTOCOL |

If G3 fails: **stop and report the numbers honestly** (the design predicts modest gains; a clean null is a valid, documentable outcome — mirror Phase 1's "documented reason why not" convention). Do not tune on 24→25 to rescue it.

---

### Task 1: Per-PA measurement frame (TDD)

**Files:**
- Create: `src/talent2.py`
- Test: `tests/test_talent2.py`

- [ ] **Step 1.1: Write the failing tests**

`tests/test_talent2.py`:
```python
import numpy as np
import polars as pl

from src.talent2 import build_pa_measurements


def _pitches():
    # 3 players' PAs, 2024. Rows: BBE tracked, BBE tracked (barrel), walk,
    # BBE untracked (null EV/LSA -> excluded from peripherals but keeps its value),
    # strikeout, non-PA pitch (dropped).
    return pl.DataFrame({
        "batter":       [1,    1,    1,    2,    2,    1],
        "game_year":    [2024, 2024, 2024, 2024, 2024, 2024],
        "type":         ["X",  "X",  "B",  "X",  "S",  "S"],
        "launch_speed": [101.3, 88.0, None, None, None, None],
        "launch_speed_angle": [6.0, 3.0, None, None, None, None],
        "estimated_woba_using_speedangle": [1.2, 0.1, None, 0.8, None, None],
        "woba_value":   [1.25, 0.0, 0.69, 0.9, 0.0, 0.0],
        "woba_denom":   [1,    1,    1,    1,    1,    None],
    })


def test_build_pa_measurements_values_match_phase1_logic():
    pam = build_pa_measurements(_pitches())
    # non-PA row dropped: 5 rows; value logic identical to talent.build_pa_values
    assert pam.height == 5
    v1 = pam.filter(pl.col("batter") == 1)["value"].sort().to_list()
    assert v1 == [0.1, 0.69, 1.2]
    assert set(pam.columns) == {"batter", "season", "value", "denom", "ev", "barrel"}


def test_build_pa_measurements_ev_barrel_only_on_tracked_bbe():
    pam = build_pa_measurements(_pitches()).sort("batter", "value")
    p1 = pam.filter(pl.col("batter") == 1).sort("value")
    # walk row: ev/barrel null; tracked BBE rows carry ev and barrel = (lsa == 6)
    assert p1.filter(pl.col("value") == 0.69)["ev"][0] is None
    assert p1.filter(pl.col("value") == 1.2)["ev"][0] == 101.3
    assert p1.filter(pl.col("value") == 1.2)["barrel"][0] == 1.0
    assert p1.filter(pl.col("value") == 0.1)["barrel"][0] == 0.0
    # player 2's BBE has null launch_speed/lsa -> untracked: ev AND barrel null,
    # but the PA still contributes its xwOBA value
    p2x = pam.filter((pl.col("batter") == 2) & (pl.col("value") == 0.8))
    assert p2x["ev"][0] is None and p2x["barrel"][0] is None
```

- [ ] **Step 1.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.talent2'`

- [ ] **Step 1.3: Implement `src/talent2.py` (module docstring + first function)**

```python
"""Level-2 true-talent xwOBA: a joint MVN measurement model over (raw xwOBA,
average exit velocity, barrel rate) per batter-season
(spec: docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md).

Phase 1 (src/talent.py) shrinks raw xwOBA toward the season league mean. Level 2
upgrades the prior: the three stats are jointly noisy measurements of correlated
latent talents, z_i ~ N((theta_i, xi_i), S_i), (theta_i, xi_i) ~ N(mu_season,
Sigma_talent). The per-player measurement covariance S_i is bootstrapped from the
player's own PAs; its OFF-DIAGONALS carry the shared sampling noise (all three
stats come from the same balls) — modeling them explicitly is what keeps the
low-PA gains honest. Posterior E[theta|z] leans on the fast-stabilizing
peripherals exactly when the xwOBA sample is small, and reduces to Phase 1 when
the peripheral dims are dropped. Pure functions only — orchestration lives in
scripts/run_talent2.py."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize

Z90 = 1.6448536269514722          # 90% two-sided normal quantile (as src/talent.py)
DIMS = ("xwoba", "avg_ev", "barrel_rate")
MIN_BBE = 5                       # fewer tracked BBE -> 1-D fallback (peripherals carry ~nothing)
FLOOR_SD_PER_PA = 0.25            # xwOBA meas-variance floor = (0.25)^2/n  (NOTES.md limitation 3)


def build_pa_measurements(pitches: pl.DataFrame) -> pl.DataFrame:
    """One row per plate appearance: (batter, season, value, denom, ev, barrel).
    value/denom exactly as talent.build_pa_values; ev = launch_speed and
    barrel = (launch_speed_angle == 6) only on tracked BBE (type X with non-null
    launch_speed AND launch_speed_angle — the ~0.3% untracked BBE keep their
    xwOBA value but are excluded from the peripheral denominators)."""
    tracked = (
        (pl.col("type") == "X")
        & pl.col("launch_speed").is_not_null()
        & pl.col("launch_speed_angle").is_not_null()
    )
    return (
        pitches.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value")),
            ev=pl.when(tracked).then(pl.col("launch_speed")),
            barrel=pl.when(tracked).then(
                (pl.col("launch_speed_angle") == 6).cast(pl.Float64)
            ),
        )
        .select("batter", season="game_year", value="value", denom="woba_denom",
                ev="ev", barrel="barrel")
    )
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: 2 passed

- [ ] **Step 1.5: Run the full suite (no regressions), then commit**

Run: `.venv/bin/pytest`
Expected: 36 passed (34 existing + 2 new)

```bash
git add src/talent2.py tests/test_talent2.py
git commit -m "feat(talent2): per-PA measurement frame with EV + barrel on tracked BBE"
```

---

### Task 2: Player measurement triples + bootstrap covariance S_i (TDD)

**Files:**
- Modify: `src/talent2.py`
- Test: `tests/test_talent2.py`

The measurement covariance is the load-bearing piece: resample each player's PAs
with replacement, recompute the triple per replicate, take the covariance. This
captures the xwOBA/EV/barrel **cross-covariances** (shared sampling noise) exactly,
and handles the mixed denominators (xwOBA is per-PA; EV/barrel are per-tracked-BBE)
with no analytic approximation.

- [ ] **Step 2.1: Write the failing tests (append to `tests/test_talent2.py`)**

```python
from src.talent2 import (
    FLOOR_SD_PER_PA,
    assemble_measurements,
    bootstrap_S,
    player_measurements,
)


def _rng():
    return np.random.default_rng(7)


def test_player_measurements_triple_and_dropped_singleton():
    pam = build_pa_measurements(_pitches())
    # add a single-PA player (no sd -> se2 null -> dropped, like Phase 1)
    single = pl.DataFrame({"batter": [9], "season": [2024], "value": [0.3],
                           "denom": [1], "ev": [90.0], "barrel": [0.0]})
    meas = player_measurements(pl.concat([pam, single.cast(pam.schema)]))
    assert 9 not in meas["batter"].to_list()
    r1 = meas.filter(pl.col("batter") == 1).row(0, named=True)
    assert r1["n"] == 3 and r1["n_bbe"] == 2
    assert abs(r1["avg_ev"] - (101.3 + 88.0) / 2) < 1e-9
    assert abs(r1["barrel_rate"] - 0.5) < 1e-9
    assert abs(r1["xwoba_raw"] - (1.2 + 0.1 + 0.69) / 3) < 1e-9
    assert r1["se2"] > 0
    # player 2: BBE untracked -> n_bbe == 0, peripherals null
    r2 = meas.filter(pl.col("batter") == 2).row(0, named=True)
    assert r2["n_bbe"] == 0 and r2["avg_ev"] is None


def test_bootstrap_S_recovers_known_covariance():
    # Player with n=4000 iid PAs where value and ev share noise by construction.
    rng = _rng()
    n = 4000
    common = rng.normal(0, 1.0, n)
    ev = 89.0 + 5.0 * common + rng.normal(0, 3.0, n)
    value = np.clip(0.32 + 0.20 * common + rng.normal(0, 0.40, n), 0, 2.0)
    barrel = (rng.random(n) < 0.05 + 0.04 * (common > 1)).astype(float)
    denom = np.ones(n)
    S = bootstrap_S(value, denom, ev, barrel, B=800, rng=_rng())
    # diagonal ~ Var(stat of the mean): Var(value)/n etc. (within 25% — B noise)
    assert abs(S[0, 0] / (value.var(ddof=1) / n) - 1) < 0.25
    assert abs(S[1, 1] / (ev.var(ddof=1) / n) - 1) < 0.25
    # shared noise -> positive xwOBA/EV cross-covariance, right magnitude
    expected_cov = np.cov(value, ev)[0, 1] / n
    assert S[0, 1] > 0 and abs(S[0, 1] / expected_cov - 1) < 0.35


def test_bootstrap_S_xwoba_var_matches_analytic_se2():
    rng = _rng()
    n = 500
    value = rng.normal(0.32, 0.45, n)
    S = bootstrap_S(value, np.ones(n), np.full(n, np.nan), np.full(n, np.nan),
                    B=800, rng=_rng())
    # peripherals absent -> those entries NaN, xwOBA var still valid
    assert np.isnan(S[1, 1]) and np.isnan(S[2, 2])
    analytic = value.var(ddof=1) / n
    assert abs(S[0, 0] / analytic - 1) < 0.2


def test_bootstrap_S_floor_on_degenerate_values():
    # identical values -> zero bootstrap variance -> floored, not 0 (NOTES lim. 3)
    n = 8
    S = bootstrap_S(np.full(n, 0.7), np.ones(n), np.full(n, np.nan),
                    np.full(n, np.nan), B=200, rng=_rng())
    assert S[0, 0] >= FLOOR_SD_PER_PA ** 2 / n


def test_assemble_measurements_aligns_and_flags():
    rng = _rng()
    rows = []
    for batter, n_pa in ((1, 200), (2, 40), (3, 3)):
        for _ in range(n_pa):
            bbe = rng.random() < 0.7
            rows.append({
                "batter": batter, "game_year": 2024, "type": "X" if bbe else "S",
                "launch_speed": float(rng.normal(89, 6)) if bbe else None,
                "launch_speed_angle": float(rng.integers(1, 7)) if bbe else None,
                "estimated_woba_using_speedangle": float(np.clip(rng.normal(0.35, 0.4), 0, 2)) if bbe else None,
                "woba_value": 0.0, "woba_denom": 1,
            })
    meas, S = assemble_measurements(build_pa_measurements(pl.DataFrame(rows)),
                                    B=200, seed=1)
    assert S.shape == (meas.height, 3, 3) and meas["s_ok"].all()
    # rows are sorted (batter, season) and S is row-aligned: bigger sample ->
    # smaller xwOBA measurement variance
    m = {b: S[i, 0, 0] for i, b in enumerate(meas["batter"].to_list())}
    assert m[1] < m[2] < m[3]
```

- [ ] **Step 2.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: new tests FAIL — `ImportError` (names not defined)

- [ ] **Step 2.3: Implement (append to `src/talent2.py`)**

```python
def player_measurements(pam: pl.DataFrame) -> pl.DataFrame:
    """Per (batter, season): the observed triple z = (xwoba_raw, avg_ev,
    barrel_rate), sample sizes, and Phase-1's analytic se2 for the xwOBA dim
    (used as cross-check and as the 1-D fallback measurement variance).
    Single-PA seasons (no sample sd) are dropped, exactly like Phase 1."""
    return (
        pam.group_by("batter", "season")
        .agg(
            PA=pl.col("denom").sum().cast(pl.Int64),
            n=pl.len(),
            num=pl.col("value").sum(),
            den=pl.col("denom").sum(),
            sd=pl.col("value").std(ddof=1),
            n_bbe=pl.col("ev").count().cast(pl.Int64),
            avg_ev=pl.col("ev").mean(),
            barrel_rate=pl.col("barrel").mean(),
        )
        .with_columns(
            xwoba_raw=pl.col("num") / pl.col("den"),
            se2=(pl.col("sd") / pl.col("n").sqrt()) ** 2,
        )
        .drop("num", "den", "sd")
        .filter(pl.col("se2").is_not_null())
        .sort("batter", "season")
    )


def bootstrap_S(value: np.ndarray, denom: np.ndarray, ev: np.ndarray,
                barrel: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Measurement covariance of the observed triple for ONE player-season, by
    resampling their PAs with replacement. ev/barrel are NaN on non-BBE rows;
    replicates are computed with NaN-aware means, and replicates with zero
    tracked BBE get NaN peripherals. Entries touching a peripheral are NaN when
    fewer than B/2 replicates were valid (caller falls back to 1-D). The xwOBA
    variance is floored at FLOOR_SD_PER_PA^2/n so degenerate tiny samples cannot
    claim certainty (Phase-1 NOTES limitation 3)."""
    n = len(value)
    idx = rng.integers(0, n, size=(B, n))
    v, d, e, b = value[idx], denom[idx], ev[idx], barrel[idx]
    den = d.sum(axis=1)
    xw = np.where(den > 0, v.sum(axis=1) / np.maximum(den, 1e-12), np.nan)
    ecnt = np.isfinite(e).sum(axis=1)
    ev_rep = np.where(ecnt > 0, np.nansum(e, axis=1) / np.maximum(ecnt, 1), np.nan)
    br_rep = np.where(ecnt > 0, np.nansum(b, axis=1) / np.maximum(ecnt, 1), np.nan)

    S = np.full((3, 3), np.nan)
    ok_x = np.isfinite(xw)
    if ok_x.sum() >= B // 2:
        S[0, 0] = xw[ok_x].var(ddof=1)
    ok3 = ok_x & np.isfinite(ev_rep) & np.isfinite(br_rep)
    if ok3.sum() >= B // 2:
        S = np.cov(np.stack([xw[ok3], ev_rep[ok3], br_rep[ok3]]), ddof=1)
    if np.isfinite(S[0, 0]):
        S[0, 0] = max(S[0, 0], FLOOR_SD_PER_PA ** 2 / n)   # raises an eigenvalue: stays PSD
    for k in (1, 2):
        if np.isfinite(S[k, k]):
            S[k, k] = max(S[k, k], 1e-8)
    return S


def assemble_measurements(pam: pl.DataFrame, B: int = 500,
                          seed: int = 20260719) -> tuple[pl.DataFrame, np.ndarray]:
    """player_measurements plus a row-aligned stack of bootstrap covariances
    S (n, 3, 3). Adds s_ok = the xwOBA variance is finite (bootstrap succeeded).
    Computed once and reused across model variants (full/ablations/diagnostic)."""
    meas = player_measurements(pam)
    lists = (
        pam.group_by("batter", "season")
        .agg(pl.col("value"), pl.col("denom"), pl.col("ev"), pl.col("barrel"))
        .sort("batter", "season")
        .join(meas.select("batter", "season"), on=["batter", "season"], how="inner")
        .sort("batter", "season")
    )
    assert lists.height == meas.height
    rng = np.random.default_rng(seed)
    S = np.empty((meas.height, 3, 3))
    for i, row in enumerate(lists.iter_rows(named=True)):
        S[i] = bootstrap_S(
            np.asarray(row["value"], float), np.asarray(row["denom"], float),
            np.asarray(row["ev"], float), np.asarray(row["barrel"], float),
            B=B, rng=rng,
        )
    return meas.with_columns(s_ok=pl.Series(np.isfinite(S[:, 0, 0]))), S
```

Note: `np.asarray(row["ev"], float)` turns polars nulls into NaN — exactly what
`bootstrap_S` expects.

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: 7 passed

- [ ] **Step 2.5: Full suite, then commit**

Run: `.venv/bin/pytest`
Expected: 41 passed

```bash
git add src/talent2.py tests/test_talent2.py
git commit -m "feat(talent2): player measurement triples + bootstrap covariance S_i with shared-noise off-diagonals"
```

---

### Task 3: Hyperparameter MLE + conditional posterior (TDD)

**Files:**
- Modify: `src/talent2.py`, `requirements.txt`
- Test: `tests/test_talent2.py`

Marginal likelihood: `z_i ~ N(mu[season_i], Sigma + S_i)` — μ per season, Σ shared,
Σ parameterized by its Cholesky factor (log-diagonal) so it is PSD by construction.
Inputs are standardized by the caller (Task 4); these functions assume O(1) scales.

- [ ] **Step 3.1: Add `scipy>=1.10` to `requirements.txt`** (after `pyarrow`), verify import:

Run: `.venv/bin/python -c "import scipy; print(scipy.__version__)"`
Expected: prints a version ≥ 1.10 (1.18.0 currently installed)

- [ ] **Step 3.2: Write the failing tests (append)**

```python
from src.talent import eb_fit, eb_shrink
from src.talent2 import mvn_mle, mvn_posterior


def _simulate_mvn(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    mu_true = np.array([[0.0, 0.2, -0.1], [0.3, -0.2, 0.1]])   # 2 seasons x 3 dims
    sd = np.array([1.0, 0.8, 0.6])
    C = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]])
    Sigma_true = C * np.outer(sd, sd)
    t = rng.integers(0, 2, n)
    theta = mu_true[t] + rng.multivariate_normal(np.zeros(3), Sigma_true, n)
    S = np.empty((n, 3, 3))
    z = np.empty((n, 3))
    for i in range(n):
        s_sd = rng.uniform(0.3, 1.5, 3)
        Si = np.diag(s_sd ** 2)
        Si[0, 1] = Si[1, 0] = 0.5 * s_sd[0] * s_sd[1]       # shared noise
        S[i] = Si
        z[i] = theta[i] + rng.multivariate_normal(np.zeros(3), Si)
    return z, S, t, mu_true, Sigma_true


def test_mvn_mle_parameter_recovery():
    z, S, t, mu_true, Sigma_true = _simulate_mvn()
    mu, Sigma = mvn_mle(z, S, t, n_seasons=2)
    assert np.abs(mu - mu_true).max() < 0.08
    assert np.abs(np.diag(Sigma) / np.diag(Sigma_true) - 1).max() < 0.15
    corr = Sigma / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))
    corr_true = Sigma_true / np.sqrt(np.outer(np.diag(Sigma_true), np.diag(Sigma_true)))
    assert np.abs(corr - corr_true).max() < 0.12


def test_mvn_mle_1d_matches_eb_fit():
    # Same setup as Phase 1's test_eb_fit_recovers_hyperparameters
    rng = np.random.default_rng(0)
    n = 4000
    theta = rng.normal(0.32, 0.05, n)
    se = rng.uniform(0.02, 0.08, n)
    raw = theta + rng.normal(0, se)
    mu, Sigma = mvn_mle(raw[:, None], (se ** 2)[:, None, None],
                        np.zeros(n, int), n_seasons=1)
    mu_eb, tau2_eb = eb_fit(raw, se ** 2)
    assert abs(mu[0, 0] - mu_eb) < 0.003
    assert abs(np.sqrt(Sigma[0, 0]) - np.sqrt(tau2_eb)) < 0.005


def test_mvn_posterior_1d_equals_eb_shrink():
    mu_s, tau2 = 0.320, 0.05 ** 2
    raw = np.array([0.40, 0.40, 0.25])
    se2 = np.array([0.02 ** 2, 0.08 ** 2, 0.03 ** 2])
    theta, var0 = mvn_posterior(raw[:, None], se2[:, None, None],
                                np.array([[mu_s]]), np.array([[tau2]]),
                                np.zeros(3, int))
    t_eb, pv_eb, *_ = eb_shrink(raw, se2, mu_s, tau2)
    assert np.allclose(theta[:, 0], t_eb) and np.allclose(var0, pv_eb)


def test_mvn_posterior_peripheral_pull_and_information_gain():
    # Two low-PA players, identical league-average xwOBA measurement; one has
    # elite, precisely-measured peripherals. Positive talent correlations must
    # pull his xwOBA talent above the mean, with LOWER posterior variance than
    # the 1-D shrink of the same xwOBA measurement.
    mu = np.array([[0.0, 0.0, 0.0]])
    C = np.array([[1.0, 0.7, 0.6], [0.7, 1.0, 0.4], [0.6, 0.4, 1.0]])
    z = np.array([[0.0, 2.0, 2.0],     # league xwOBA, elite peripherals
                  [0.0, 0.0, 0.0]])    # league everything
    S = np.tile(np.diag([4.0, 0.1, 0.1]), (2, 1, 1))   # xwOBA noisy, periphs tight
    theta, var0 = mvn_posterior(z, S, mu, C, np.zeros(2, int))
    assert theta[0, 0] > 0.5 and abs(theta[1, 0]) < 1e-9
    t1, v1 = mvn_posterior(z[:, :1], S[:, :1, :1], mu[:, :1], C[:1, :1],
                           np.zeros(2, int))
    assert var0[0] < v1[0]             # peripherals resolve xwOBA-talent variance
```

- [ ] **Step 3.3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: new tests FAIL — `ImportError`

- [ ] **Step 3.4: Implement (append)**

```python
def mvn_mle(z: np.ndarray, S: np.ndarray, season_idx: np.ndarray,
            n_seasons: int) -> tuple[np.ndarray, np.ndarray]:
    """Marginal MLE of per-season means mu (n_seasons, D) and the shared talent
    covariance Sigma (D, D) under z_i ~ N(mu[t_i], Sigma + S_i). Sigma is
    parameterized by its Cholesky factor with log-diagonal (PSD by construction).
    Assumes standardized inputs (O(1) scales). L-BFGS-B, numeric gradient — 18
    params at D=3, a few seconds on ~2k rows."""
    n, D = z.shape
    tril = np.tril_indices(D)

    def build_L(lp: np.ndarray) -> np.ndarray:
        L = np.zeros((D, D))
        L[tril] = lp
        L[np.diag_indices(D)] = np.exp(np.diag(L))
        return L

    def nll(params: np.ndarray) -> float:
        mu = params[: n_seasons * D].reshape(n_seasons, D)
        Sigma = (L := build_L(params[n_seasons * D:])) @ L.T
        C = Sigma[None] + S
        diff = z - mu[season_idx]
        sol = np.linalg.solve(C, diff[..., None])[..., 0]
        _, logdet = np.linalg.slogdet(C)
        return 0.5 * float(np.sum(logdet + np.einsum("nd,nd->n", diff, sol)))

    # init: per-season means; Sigma0 = cov(z) - mean(S), eigenvalue-clipped PSD
    mu0 = np.stack([z[season_idx == t].mean(axis=0) if (season_idx == t).any()
                    else z.mean(axis=0) for t in range(n_seasons)])
    Sigma0 = np.cov(z, rowvar=False).reshape(D, D) - S.mean(axis=0)
    w, V = np.linalg.eigh((Sigma0 + Sigma0.T) / 2)
    L0 = np.linalg.cholesky(V @ np.diag(np.clip(w, 1e-4, None)) @ V.T)
    lp0 = L0[tril].copy()
    lp0[np.cumsum(np.arange(1, D + 1)) - 1] = np.log(np.diag(L0))
    x0 = np.concatenate([mu0.ravel(), lp0])
    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 500})
    mu = res.x[: n_seasons * D].reshape(n_seasons, D)
    L = build_L(res.x[n_seasons * D:])
    return mu, L @ L.T


def mvn_posterior(z: np.ndarray, S: np.ndarray, mu: np.ndarray,
                  Sigma: np.ndarray, season_idx: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form conditional posterior of the latent talent vector per row:
    theta = mu + Sigma (Sigma+S_i)^-1 (z - mu); V = Sigma - Sigma (Sigma+S_i)^-1 Sigma.
    Returns (theta (n, D), posterior variance of dim 0 (n,)). At D=1 this is
    exactly Phase 1's eb_shrink."""
    C = Sigma[None] + S
    A = np.transpose(np.linalg.solve(C, np.broadcast_to(Sigma, C.shape).copy()),
                     (0, 2, 1))                       # Sigma (Sigma+S_i)^-1
    diff = z - mu[season_idx]
    theta = mu[season_idx] + np.einsum("nij,nj->ni", A, diff)
    V0 = Sigma[0, 0] - np.einsum("nj,j->n", A[:, 0, :], Sigma[:, 0])
    return theta, np.maximum(V0, 0.0)
```

- [ ] **Step 3.5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: 11 passed (watch runtime — the two MLE tests should stay under ~15 s total)

- [ ] **Step 3.6: Full suite, then commit**

```bash
git add src/talent2.py tests/test_talent2.py requirements.txt
git commit -m "feat(talent2): marginal-MLE hyperparameters + closed-form MVN conditional posterior"
```

---

### Task 4: Table assembly with standardization and 1-D fallback (TDD)

**Files:**
- Modify: `src/talent2.py`
- Test: `tests/test_talent2.py`

- [ ] **Step 4.1: Write the failing tests (append)**

```python
from src.talent import build_pa_values, build_talent_table
from src.talent2 import MIN_BBE, build_talent2_table


def _synthetic_league(n_players=150, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for batter in range(n_players):
        talent_ev = rng.normal(89, 3)
        talent_x = 0.31 + 0.012 * (talent_ev - 89) + rng.normal(0, 0.02)
        n_pa = int(rng.integers(4, 500))     # low end guarantees some n_bbe < MIN_BBE

        for _ in range(n_pa):
            bbe = rng.random() < 0.7
            ev = rng.normal(talent_ev, 7) if bbe else None
            val = float(np.clip(rng.normal(talent_x, 0.45), 0, 2)) if bbe else \
                  float(rng.random() < 0.3) * 0.7
            rows.append({
                "batter": batter, "game_year": 2024, "type": "X" if bbe else "S",
                "launch_speed": ev,
                "launch_speed_angle": (6.0 if (bbe and ev > 99) else 3.0) if bbe else None,
                "estimated_woba_using_speedangle": val if bbe else None,
                "woba_value": val, "woba_denom": 1,
            })
    return pl.DataFrame(rows)


def test_build_talent2_table_structure_and_fallback():
    pitches = _synthetic_league()
    pam = build_pa_measurements(pitches)
    meas, S = assemble_measurements(pam, B=200, seed=2)
    tbl, hypers = build_talent2_table(meas, S, fit_min_pa=100)
    need = {"batter", "season", "PA", "n_bbe", "xwoba_raw", "avg_ev",
            "barrel_rate", "xwoba_talent2", "talent2_var", "talent2_lo",
            "talent2_hi", "reliability2", "used_dims"}
    assert need.issubset(tbl.columns) and tbl.height == meas.height
    assert set(tbl["used_dims"].unique().to_list()) <= {"3d", "1d"}
    # the fallback path must actually be exercised, not vacuously true
    tiny = tbl.filter(pl.col("n_bbe") < MIN_BBE)
    assert tiny.height > 0 and tiny["used_dims"].eq("1d").all()
    assert tbl["talent2_lo"].lt(tbl["talent2_hi"]).all()
    assert 0.25 < hypers["mu"][0][0] < 0.40          # xwOBA league mean, unstd
    # positive xwOBA/EV talent correlation was built in -> recovered sign
    assert hypers["Sigma"][0][1] > 0


def test_build_talent2_table_1d_matches_phase1():
    pitches = _synthetic_league()
    pam = build_pa_measurements(pitches)
    meas, S = assemble_measurements(pam, B=400, seed=2)
    tbl, _ = build_talent2_table(meas, S, dims=("xwoba",), fit_min_pa=100)
    p1 = build_talent_table(build_pa_values(pitches), fit_min_pa=100)
    j = tbl.join(p1.select("batter", "season", "xwoba_talent"),
                 on=["batter", "season"], how="inner")
    assert j.height == tbl.height
    d = (j["xwoba_talent2"] - j["xwoba_talent"]).abs()
    r = np.corrcoef(j["xwoba_talent2"].to_numpy(), j["xwoba_talent"].to_numpy())[0, 1]
    assert r > 0.995 and d.median() < 0.005


def test_build_talent2_table_peripheral_pull_at_low_pa():
    # In the synthetic league, xwOBA talent tracks EV talent. Among LOW-PA
    # players, the 3-D posterior must move high-EV players up relative to the
    # 1-D (Phase-1-style) shrink of the same xwOBA measurement.
    pitches = _synthetic_league(seed=13)
    pam = build_pa_measurements(pitches)
    meas, S = assemble_measurements(pam, B=200, seed=2)
    t3, _ = build_talent2_table(meas, S, fit_min_pa=100)
    t1, _ = build_talent2_table(meas, S, dims=("xwoba",), fit_min_pa=100)
    j = (t3.select("batter", "PA", "avg_ev", "n_bbe", t2=pl.col("xwoba_talent2"))
           .join(t1.select("batter", t1c=pl.col("xwoba_talent2")), on="batter")
           .filter((pl.col("PA") < 80) & (pl.col("n_bbe") >= 5)))
    hi = j.filter(pl.col("avg_ev") > j["avg_ev"].quantile(0.8))
    lo = j.filter(pl.col("avg_ev") < j["avg_ev"].quantile(0.2))
    assert (hi["t2"] - hi["t1c"]).mean() > (lo["t2"] - lo["t1c"]).mean()
```

- [ ] **Step 4.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_talent2_table'`

- [ ] **Step 4.3: Implement (append)**

```python
def build_talent2_table(meas: pl.DataFrame, S_all: np.ndarray,
                        dims: tuple[str, ...] = DIMS, fit_min_pa: int = 100,
                        zero_offdiag: bool = False,
                        fit_seasons: list[int] | None = None
                        ) -> tuple[pl.DataFrame, dict]:
    """Assemble the Level-2 talent table. Hyperparameters (per-season mu, shared
    Sigma) are fit by marginal MLE on the stable population (PA >= fit_min_pa,
    valid measurements), standardized per dim; posteriors are computed for every
    row. Rows without usable peripherals (n_bbe < MIN_BBE, missing values, or a
    failed bootstrap) fall back to the 1-D xwOBA-only model with the analytic
    se2 (floored) — i.e., exactly Phase 1's machinery. zero_offdiag drops the
    S_i off-diagonals (the shared-noise diagnostic; NOT for production).
    fit_seasons restricts the HYPERPARAMETER fit to those seasons (posteriors
    still computed for all rows) — the leakage-sensitivity check; default None
    fits on all seasons, matching Phase 1's convention."""
    assert dims[0] == "xwoba"
    d_idx = [DIMS.index(d) for d in dims]
    D = len(d_idx)
    seasons = sorted(meas["season"].unique().to_list())
    t_idx = np.array([seasons.index(s) for s in meas["season"].to_list()])
    z = meas.select("xwoba_raw", "avg_ev", "barrel_rate").to_numpy()[:, d_idx]
    S = S_all[:, d_idx][:, :, d_idx].copy()
    if zero_offdiag:
        S = S * np.eye(D)[None]

    ok = meas["s_ok"].to_numpy() & np.isfinite(z).all(axis=1) \
        & np.isfinite(S).all(axis=(1, 2))
    if D > 1:
        ok &= meas["n_bbe"].to_numpy() >= MIN_BBE

    pa = meas["PA"].to_numpy()
    fit = ok & (pa >= fit_min_pa)
    if fit_seasons is not None:
        fit &= np.isin(meas["season"].to_numpy(), fit_seasons)
    center = z[fit].mean(axis=0)
    scale = z[fit].std(axis=0, ddof=1)
    assert (scale > 0).all()
    zs = (z - center) / scale
    Ss = S / np.outer(scale, scale)[None]

    mu, Sigma = mvn_mle(zs[fit], Ss[fit], t_idx[fit], len(seasons))

    theta0 = np.empty(len(meas))
    var0 = np.empty(len(meas))
    th, v = mvn_posterior(zs[ok], Ss[ok], mu, Sigma, t_idx[ok])
    theta0[ok], var0[ok] = th[:, 0], v

    if (~ok).any():                     # 1-D fallback on the analytic se2, floored
        n_arr = meas["n"].to_numpy().astype(float)
        se2 = np.maximum(np.nan_to_num(meas["se2"].to_numpy(), nan=np.inf),
                         FLOOR_SD_PER_PA ** 2 / n_arr)
        z1 = (meas["xwoba_raw"].to_numpy()[~ok, None] - center[0]) / scale[0]
        S1 = (se2[~ok] / scale[0] ** 2)[:, None, None]
        th1, v1 = mvn_posterior(z1, S1, mu[:, :1], Sigma[:1, :1], t_idx[~ok])
        theta0[~ok], var0[~ok] = th1[:, 0], v1

    talent = center[0] + theta0 * scale[0]
    var = var0 * scale[0] ** 2
    half = Z90 * np.sqrt(var)
    hypers = {
        "dims": list(dims), "seasons": seasons,
        "mu": (center + mu * scale).tolist(),
        "Sigma": (Sigma * np.outer(scale, scale)).tolist(),
        "center": center.tolist(), "scale": scale.tolist(),
        "n_fit": int(fit.sum()), "n_fallback_1d": int((~ok).sum()),
        "zero_offdiag": zero_offdiag, "fit_seasons": fit_seasons,
    }
    tbl = meas.with_columns(
        xwoba_talent2=pl.Series(talent),
        talent2_var=pl.Series(var),
        talent2_lo=pl.Series(talent - half),
        talent2_hi=pl.Series(talent + half),
        reliability2=pl.Series(1.0 - var0 / Sigma[0, 0]),
        used_dims=pl.Series(np.where(ok, "3d" if D > 1 else "1d", "1d")),
    )
    return tbl, hypers
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_talent2.py -v`
Expected: 14 passed

- [ ] **Step 4.5: Full suite, then commit**

```bash
git add src/talent2.py tests/test_talent2.py
git commit -m "feat(talent2): Level-2 table assembly — standardized MLE, closed-form posteriors, 1-D fallback"
```

---

### Task 5: Runner part 1 — L2a regression gates on real data

**Files:**
- Create: `scripts/run_talent2.py`

No new statistics here — this proves the new machinery reproduces Phase 1 before
adding anything. **Gates G1 and G2 from the table above.**

- [ ] **Step 5.1: Implement the runner skeleton + L2a stage**

`scripts/run_talent2.py` (complete file for this task; Task 6 replaces the one
marked placeholder line with the real `stage_full` call):
```python
"""Level-2 talent model runner (spec: 2026-07-19 phase2 design response; plan:
2026-07-19-xwobart-phase2-level2-talent). Stage 'l2a' proves the joint-MVN
machinery reproduces Phase 1 when restricted to xwOBA only (gates G1/G2);
stage 'full' fits the 3-D model and runs the validation races (G3-G6).
Run from repo root: `.venv/bin/python scripts/run_talent2.py [--stage l2a|full]`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.config import load_config
from src.talent2 import DIMS, assemble_measurements, build_pa_measurements, build_talent2_table
from benchmark_vs_savant import actual_woba, _calibrated_rmse, _pearson
from run_talent import load_pitches

C_TAL = "#4878CF"; C_TAL2 = "#EE854A"; C_REF = "#8a8a8a"

# Frozen Phase-1 anchors (results/talent/talent_metrics.json, 2026-07-18)
P1_R_PA100, P1_R_PA30, P1_N_ROWS = 0.4886, 0.4669, 2636
BOOT_B, BOOT_SEED = 500, 20260719
PREDS = ["xwoba_talent2", "xwoba_talent", "xwoba_raw", "xwoba_savant"]


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} — {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def build_measurements(cfg) -> tuple[pl.DataFrame, np.ndarray]:
    """Measurement triples + bootstrap covariances, with a row index (ridx) so
    joined/filtered frames can always index back into the S stack."""
    pam = build_pa_measurements(load_pitches(cfg, cfg.all_seasons))
    meas, S = assemble_measurements(pam, B=BOOT_B, seed=BOOT_SEED)
    return meas.with_row_index("ridx"), S


def phase1_cols(cfg) -> pl.DataFrame:
    """Frozen Phase-1 columns, renamed to avoid collisions with the L2 table."""
    p1 = pl.read_parquet(cfg.results_dir / "talent" / "talent_table.parquet")
    return p1.select(
        "batter", "season", "xwoba_talent", "xwoba_savant", "player_name",
        se2_p1="se2", p1_lo="talent_lo", p1_hi="talent_hi",
    )


def with_targets(tbl: pl.DataFrame, cfg) -> pl.DataFrame:
    act = actual_woba(cfg.raw_dir, cfg.all_seasons)
    return tbl.join(act.select("batter", "season", "actual_woba"),
                    on=["batter", "season"], how="inner")


def make_pairs(base: pl.DataFrame, seasons_t: list[int], pa_t_floor: int,
               min_pa_next: int) -> pl.DataFrame:
    """Season-T rows (PA >= pa_t_floor) joined to their T+1 actual wOBA, keeping
    players with a stable next-season sample (pa_next >= min_pa_next). Same
    logic as run_talent.validate's make_pairs, parameterized by season list so
    the select (22->23, 23->24) / confirm (24->25) split reuses it."""
    rows = []
    for t in seasons_t:
        a = base.filter(pl.col("season") == t)
        b = base.filter(pl.col("season") == t + 1).select(
            "batter", target="actual_woba", pa_next="PA")
        rows.append(
            a.join(b, on="batter", how="inner")
            .filter((pl.col("PA") >= pa_t_floor) & (pl.col("pa_next") >= min_pa_next))
            .with_columns(season_t=pl.lit(t))
        )
    return pl.concat(rows)


def race(pairs: pl.DataFrame, preds: list[str]) -> dict:
    """{pred: {r, rmse_calibrated}} + n for each predictor column."""
    tgt = pairs["target"].to_numpy()
    out = {"n": pairs.height}
    for p in preds:
        v = pairs[p].to_numpy()
        out[p] = {"r": _pearson(v, tgt), "rmse_calibrated": _calibrated_rmse(v, tgt)}
    return out


def stage_l2a(cfg, meas: pl.DataFrame, S: np.ndarray, p1: pl.DataFrame
              ) -> tuple[pl.DataFrame, dict]:
    tbl, hypers = build_talent2_table(meas, S, dims=("xwoba",), fit_min_pa=cfg.min_pa)
    j = tbl.join(p1, on=["batter", "season"], how="inner")
    gates = [
        _gate("G1.height", tbl.height == P1_N_ROWS and j.height == tbl.height,
              f"rows {tbl.height} (phase1 {P1_N_ROWS}), joined {j.height}"),
    ]
    r = _pearson(j["xwoba_talent2"].to_numpy(), j["xwoba_talent"].to_numpy())
    med = float((j["xwoba_talent2"] - j["xwoba_talent"]).abs().median())
    gates.append(_gate("G1.match", r >= 0.999 and med <= 0.002,
                       f"corr {r:.5f}, median |diff| {med:.5f}"))
    # G2: bootstrap xwOBA SE vs Phase-1 analytic SE, rows PA>=30
    k = j.filter(pl.col("PA") >= 30)
    boot_se = np.sqrt(S[k["ridx"].to_numpy(), 0, 0])
    ana_se = np.sqrt(k["se2_p1"].to_numpy())
    r_se = _pearson(boot_se, ana_se)
    ratio = float(np.median(boot_se / ana_se))
    gates.append(_gate("G2.se", r_se >= 0.98 and 0.9 <= ratio <= 1.1,
                       f"corr {r_se:.4f}, median ratio {ratio:.3f}"))
    # G1 validation anchors: same race machinery as stage_full
    base = with_targets(j, cfg)
    r100 = race(make_pairs(base, cfg.all_seasons[:-1], cfg.min_pa, cfg.min_pa),
                ["xwoba_talent2"])["xwoba_talent2"]["r"]
    r30 = race(make_pairs(base, cfg.all_seasons[:-1], 30, cfg.min_pa),
               ["xwoba_talent2"])["xwoba_talent2"]["r"]
    gates.append(_gate("G1.val100", abs(r100 - P1_R_PA100) <= 0.005,
                       f"r {r100:.4f} vs anchor {P1_R_PA100}"))
    gates.append(_gate("G1.val30", abs(r30 - P1_R_PA30) <= 0.005,
                       f"r {r30:.4f} vs anchor {P1_R_PA30}"))
    return tbl, {"gates": gates, "hypers": hypers, "r_pa100": r100, "r_pa30": r30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["l2a", "full"], default="l2a")
    args = ap.parse_args()
    cfg = load_config()
    outdir = cfg.results_dir / "talent2"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    meas, S = build_measurements(cfg)
    p1 = phase1_cols(cfg)

    metrics = {}
    tbl_a, metrics["l2a"] = stage_l2a(cfg, meas, S, p1)
    tbl_a.write_parquet(outdir / "l2a_table.parquet")
    if args.stage == "full":
        raise SystemExit("stage 'full' arrives in Task 6")   # replaced in Task 6

    (outdir / "talent2_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))
    failed = [g["name"] for st in metrics.values() for g in st["gates"] if not g["pass"]]
    print(f"  wrote {outdir}/")
    if failed:
        print(f"  HARD GATE FAILURES: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

Design notes locked in by this code: `ridx` ties every joined/filtered row back
to its `S` slice; the frozen Phase-1 table (not a re-derivation) is the
comparison object; gate failures write metrics first, then exit non-zero so an
executing agent cannot miss them.

- [ ] **Step 5.2: Run stage l2a on real data**

Run: `.venv/bin/python scripts/run_talent2.py --stage l2a`
Expected: all G1/G2 gates print PASS; exit code 0. Bootstrap step ~1–3 min
(2,636 players × B=500); MLE seconds. If G1.match fails marginally (corr
0.995–0.999), inspect in this order before touching tolerances: (1) L2a fits ONE
shared Σ across seasons while Phase 1 fits per-season τ — per-season τ is nearly
constant (0.0307–0.0323) so this should pass, but it is the likeliest structural
cause of a marginal miss (and the single-season synthetic parity test cannot
catch it); (2) MLE-vs-method-of-moments τ differences on the real, skewed
population; (3) the ratio-vs-mean estimator difference on degenerate rows.
Inspect the largest |diff| rows and document whichever it is.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/run_talent2.py results/talent2/
git commit -m "feat(talent2): runner L2a stage — joint-MVN machinery reproduces Phase 1 (gates G1/G2)"
```

---

### Task 6: Runner part 2 — L2b validation races, ablations, diagnostic, figures

**Files:**
- Modify: `scripts/run_talent2.py`

- [ ] **Step 6.1: Implement `stage_full`**

Wiring: replace the Task-5 placeholder line in `main()` with
`metrics["l2b"] = stage_full(cfg, meas, S, p1, outdir, figdir)` and flip the
argparse default to `"full"`. `stage_full` returns a dict that MUST contain a
`"gates"` list (G3/G4 entries via `_gate`) so `main()`'s hard-fail sweep covers
it; the G5 tripwire is a reported flag (`offdiag_alarm`), not a gate entry.

Structure (all pieces land in the `"l2b"` metrics key):

1. **Fit** `tbl2, hypers = build_talent2_table(meas, S, dims=DIMS, fit_min_pa=cfg.min_pa)`.
2. **Base frame:** join tbl2 + `phase1_cols` (which carries `xwoba_talent`,
   `xwoba_savant`, `player_name`, `p1_lo/p1_hi`, `se2_p1`) + `actual_woba(
   cfg.raw_dir, cfg.all_seasons)`. `xwoba_raw` comes from tbl2 itself — it is
   constructed identically to Phase 1's, so no second copy is joined. Assert
   height 2,636 after joins.
3. **Primary races** with `PREDS = ["xwoba_talent2", "xwoba_talent", "xwoba_raw",
   "xwoba_savant"]`:
   - pooled PA_T≥100 (all pairs 22→23, 23→24, 24→25): r + calibrated RMSE,
   - pooled PA_T≥30: r + calibrated RMSE,
   - by-band table (30–60, 60–100, 100–250, 250+) as Phase 1 did,
   - split: `select` = pairs with T∈{2022, 2023}; `confirm` = T=2024 — both
     reported for PA≥30 pooled (gate G6 is about *reporting*, not thresholds).
4. **Gates:** G3 = (r and rmse_calibrated better than `xwoba_talent`) on PA≥30
   all-pairs pooled; G4 = PA≥100 r within 0.005. HARD → SystemExit(1) on failure
   **after writing metrics + NOTES-ready numbers** (so the failure is documented).
5. **Paired bootstrap delta** (5,000 reps, seed 42, PA≥30 pooled): resample
   player-pair rows; Δr and Δrmse_cal for talent2 − talent; report mean, 95% CI,
   `frac_better`. Reported, not gated (design expects modest gains; the CI is
   honesty, not a pass bar).
6. **Ablations** (G6 protocol): dims `("xwoba","avg_ev")` and `("xwoba","barrel_rate")`,
   scored on the `select` split only, PA≥30 pooled r/RMSE; report side by side with
   full. If an ablation beats full on `select`, note it and report its `confirm`
   too — but the shipped table stays the full 3-D fit unless the review says otherwise.
7. **Off-diagonal tripwire (G5):** rebuild with `zero_offdiag=True`, race PA≥30
   pooled; `artifact_gap = (r_zeroed_gain) - (r_proper_gain)` where gain is vs
   `xwoba_talent`. FAIL the diagnostic (metrics flag `offdiag_alarm: true`, big
   printed warning, NOTES paragraph) if `artifact_gap > 0.005`; do not SystemExit —
   this is a finding to report, not a build error.
8. **Leakage sensitivity (spec risk-1 guard ii):** rebuild with
   `fit_seasons=cfg.train_seasons` (hyperparameters from 2022–24 only, 2025
   measurement rows excluded from the fit; posteriors unchanged in construction)
   and report the PA≥30 pooled race deltas vs the all-season fit under a
   `hypers_2224_sensitivity` key. Expect near-identical numbers; a material
   shift means the shared Σ was leaning on 2025 and the NOTES must say so.
9. **Figures** (style mirrors `run_talent.py`):
   - `figures/peripheral_pull.png`: scatter x=PA (log), y=`xwoba_talent2 −
     xwoba_talent`, colored by `barrel_rate` (viridis, colorbar). The design's
     signature: pull concentrated at low PA, signed by peripheral quality.
   - `figures/interval_width_vs_pa.png`: median 90% width by PA band (30–60,
     60–100, 100–250, 250–450, 450+), Phase 1 (`p1_hi − p1_lo`, color C_TAL) vs
     Level 2 (`talent2_hi − talent2_lo`, C_TAL2), grouped bars. Expectation:
     Level-2 narrower at the low-PA bands (peripheral information), similar high.
10. **Outputs:** `talent2_table.parquet` (full-fit table + phase-1 join columns),
   `talent2_metrics.json` (gates, races, split, bootstrap, ablations, tripwire,
   hypers incl. the unstandardized Σ_talent correlation matrix), figures.

- [ ] **Step 6.2: Run the full stage**

Run: `.venv/bin/python scripts/run_talent2.py --stage full`
Expected: gates G1–G4 PASS (G3 is the phase's thesis — if it fails, STOP per the
gate table: write everything, exit 1, report honestly). Tripwire G5 prints its
gap. Check `results/talent2/figures/*.png` render sensibly.

- [ ] **Step 6.3: Sanity-read the numbers before committing**

Read `talent2_metrics.json` and confirm, at minimum:
- `hypers.Sigma` xwOBA/EV and xwOBA/barrel talent correlations are **positive**
  (baseball says so; a negative sign means a wiring bug, likely standardization),
- `n_fallback_1d` is small (expect a few hundred at most — tiny-BBE seasons),
- per-season μ (xwOBA dim) ≈ 0.305–0.318 (Phase-1 anchors),
- the by-band table's low bands move in talent2's favor vs talent.

- [ ] **Step 6.4: Commit**

```bash
git add scripts/run_talent2.py results/talent2/
git commit -m "feat(talent2): L2b validation — races, paired bootstrap, ablations, shared-noise tripwire, figures"
```

---

### Task 7: Documentation + supersession + final verification

**Files:**
- Create: `results/talent2/NOTES.md`
- Modify: `results/RESULTS.md`, `docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md`

- [ ] **Step 7.1: Write `results/talent2/NOTES.md`** — mirror the voice and shape of
`results/talent/NOTES.md`: Goal ("shrink toward what the peripherals imply, not the
league mean"), Construction (the 4-step pipeline: measurement triple → bootstrap S_i
with off-diagonals → per-season μ + shared Σ by marginal MLE on PA≥100 → closed-form
posterior; 1-D fallback; the se² floor closing Phase-1 limitation 3), a table of the
fitted Σ_talent correlations, the validation table (talent2 vs talent vs raw vs
Savant, PA≥100 and PA≥30 pooled, with the select/confirm split), the paired-bootstrap
CI, the off-diagonal tripwire result **with one paragraph explaining why it matters**
(shared sampling noise = design risk #1), the leakage-sensitivity result (hypers are
fit on all seasons for Phase-1 parity; the 2022–24-only refit shows how much that
matters — acknowledge this convention explicitly, it deviates from the spec's letter),
and a Limitations section (single-season only — multi-season/age pooling is the known
next lever; interval still estimation-only — surface variance arrives with the
Stage-3 refit; Gaussian tails on a bounded stat).
Every number filled from `talent2_metrics.json` — no placeholders.

- [ ] **Step 7.2: Add a `results/RESULTS.md` section** "Level-2 talent model (joint
MVN over xwOBA + peripherals)" — a compact version of NOTES.md: construction in ~5
lines, the headline validation table, gates G1–G5 outcomes, reproduce command, and
pointer to the design docs + this plan.

- [ ] **Step 7.3: Update the old roadmap** `docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md` —
prepend a short status block: the two-stage contact-prior variant was empirically
killed (structural no-op, τ_resid²≈1e-4) and the event-level batter-intercept variant
rejected by the design review (models an r≈0.12 channel); Phase 2 now follows
`docs/superpowers/specs/2026-07-19-xwobart-phase2-design-response.md`, Stage 1
implemented by this plan (`2026-07-19-xwobart-phase2-level2-talent.md`); the
per-event-EV persistence prerequisite MOVES to Stage 3 (surface refit) and now
persists per-event value **draws**, not just means. Keep the original text below the
block for history.

- [ ] **Step 7.4: Final verification, then commit**

Run: `.venv/bin/pytest` — Expected: all green (48 tests: 34 existing + 14 new).
Run: `.venv/bin/python scripts/run_talent2.py --stage full` — Expected: gates PASS, idempotent outputs.

```bash
git add results/talent2/NOTES.md results/RESULTS.md docs/superpowers/plans/2026-07-18-xwobart-phase2-bart-prior.md
git commit -m "docs: Level-2 talent model results + Phase-2 roadmap supersession"
```

---

## Later stages (context only — NOT in this plan)

Per the design response: **Stage 2** cache rebuild adding `hc_x`/`hc_y`/`stand` +
pull-mirrored spray angle with sign QC; **Stage 3** one 5-feature BART surface refit
(same subsample protocol; persist ~100–200 thinned per-event value draws; ELPD vs the
−80107 anchor); **Stage 4** re-run Level 2 with the surface-draw variance added to
`S_i[0,0]`, the spray-conditioned vs spray-marginalized rollup A/B, and 50/80/90%
interval-coverage validation by PA band. Multi-season + age pooling at Level 2 is the
flagged follow-on after that. Each gets its own plan once Stage 1's numbers are in.
