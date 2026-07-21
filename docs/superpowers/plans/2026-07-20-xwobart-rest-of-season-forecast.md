# Rest-of-Season xwOBA Forecast — Implementation Plan (rung a)

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking. Follow @superpowers-extended-cc:test-driven-development for every task: write the failing test first, watch it fail, implement the minimum, watch it pass, commit.

**Goal:** Build and validate a mid-season forecaster of a hitter's **final full-season xwOBA** — a descriptor (current talent estimate) plus a calibrated range for where the season line finishes — using an EB-fit hierarchical player model that pools a hitter's own prior seasons.

**Architecture:** Reuse the Level-2 measurement machinery (`src/talent2.py`) but replace the single-season league-mean prior with a hierarchical talent structure across a player's seasons (career random intercept + iid season drift; **rung a is xwOBA-only, no aging**). Hyperparameters are fit once by empirical-Bayes marginal MLE; each (player, cutpoint) posterior is a closed-form Gaussian conditioning. The range combines that posterior with a forward-bootstrap of the remaining-PA sample, blended into the final line by the known fraction-of-season-remaining `w`. Validation stands at mid-season cutpoints and scores against the realized final line — same-season, no drift, observable.

**Tech Stack:** Python 3, numpy, polars, scipy (`minimize`, L-BFGS-B), matplotlib. Runs via `.venv/bin/python`. Tests via pytest (`tests/`, `pytest.ini` → `testpaths = tests`).

**Spec:** `docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-forecast-design.md` (read §3, §4, §5, §7–9 before starting).

**Out of scope (follow-on plans):** rung (b) peripherals (Level-2 joint MVN measurement), rung (c) aging curve + AR(1) drift (needs an external birthdate join). This plan is rung (a): the base multi-year lever + full validation harness + all five benchmarks + all five gates.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/talent3.py` (create) | Model core: PA frame w/ `game_date`, cutpoint split, per-sample measurement (reuses `bootstrap_S`), causal per-season `μ_t`, EB hyperparameter fit, closed-form per-cutpoint posterior. Pure functions only. |
| `src/forecast.py` (create) | Range construction: the final-line blend identity, the forward-bootstrap forecaster, the analytic fallback. Pure functions. |
| `src/benchmarks.py` (create) | The five benchmark point forecasts (naive, league-shrunk, single-season L2, Marcel, Savant-to-date). Pure functions. |
| `scripts/run_talent3.py` (create) | Orchestration: cutpoint sweep, LOSO hyperparameter fitting, leakage assertions, gate scoring, figures, NOTES. Mirrors `scripts/run_talent2.py`. |
| `tests/test_talent3.py` (create) | Unit + regression + leakage tests for `talent3`. |
| `tests/test_forecast.py` (create) | Blend identity, forward-bootstrap coverage/asymmetry tests. |
| `tests/test_benchmarks.py` (create) | Benchmark unit tests (Marcel weights, etc.). |
| `results/talent3/` (create, gitignored outputs) | `forecast_table.parquet`, `metrics.json`, `figures/`, `NOTES.md`. |

**Reuse (do not reimplement):** `src/talent2.py:bootstrap_S` (measurement variance, floored), `build_pa_measurements` value logic, `mvn_posterior`/`mvn_mle` (reference for the Gaussian machinery); `src/talent.py:eb_fit`/`eb_shrink` (Phase-1 league-shrunk benchmark); `run_talent.load_pitches`, `src/config.load_config`; `benchmark_vs_savant.actual_woba`/`_calibrated_rmse`/`_pearson`; `run_talent2._gate`/`paired_bootstrap` patterns.

**Key design decisions locked for this plan (from spec §14):**
- **Causal `μ_t` estimator:** the league mean rate over **all players' first-*k* PAs** in season *t* (uniform, causal, leak-free). At the full-season cutpoint this equals the plain season league mean (used by G5).
- **`w` = `D_rest / (D_obs + D_rest)`** using **realized** remaining denom (validation uses the true schedule).
- **Cutpoints:** `k ∈ {50, 100, 150, 200, 300}` PA; eligibility `D_rest ≥ 30`.
- **Rung a latent:** `θ_{i,t} = μ_t + η_i + u_{i,t}`, `η_i ~ N(0, σ_η²)`, `u_{i,t} ~ N(0, σ_u²)`. No aging term.

---

## Task 0: Final-line blend identity (`src/forecast.py`)

**Files:**
- Create: `src/forecast.py`
- Test: `tests/test_forecast.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forecast.py
import numpy as np
from src.forecast import final_line_blend


def test_blend_equals_direct_full_season_rate():
    # Two disjoint PA sets; the blend must equal the pooled Σv/Σd exactly.
    v_obs, d_obs = np.array([1.2, 0.0, 0.69]), np.array([1.0, 1.0, 1.0])
    v_rest, d_rest = np.array([0.9, 2.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0])
    r_obs, D_obs = v_obs.sum() / d_obs.sum(), d_obs.sum()
    r_rest, D_rest = v_rest.sum() / d_rest.sum(), d_rest.sum()

    direct = (v_obs.sum() + v_rest.sum()) / (d_obs.sum() + d_rest.sum())
    blend, w = final_line_blend(r_obs, D_obs, r_rest, D_rest)

    assert w == D_rest / (D_obs + D_rest)
    assert abs(blend - direct) < 1e-12
    assert abs(blend - ((1 - w) * r_obs + w * r_rest)) < 1e-12


def test_blend_w_zero_returns_locked_in():
    blend, w = final_line_blend(0.333, 500.0, 0.999, 0.0)
    assert w == 0.0 and abs(blend - 0.333) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forecast.py -v`
Expected: FAIL — `ImportError: cannot import name 'final_line_blend'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/forecast.py
"""Rest-of-season xwOBA forecast: the final-line blend and the forward-bootstrap
range. Pure functions; orchestration lives in scripts/run_talent3.py.

The final full-season line is a KNOWN-weight blend of the locked-in observed rate
and the uncertain rest-of-season rate (spec §3):
    r_final = (1 - w) * r_obs + w * r_rest,   w = D_rest / (D_obs + D_rest).
So the only quantity to model is r_rest; the forecast error is w * (r_hat_rest - r_rest)."""
from __future__ import annotations

import numpy as np


def final_line_blend(r_obs: float, D_obs: float, r_rest: float, D_rest: float
                     ) -> tuple[float, float]:
    """Final full-season rate from the locked-in observed piece and a
    rest-of-season rate. Returns (r_final, w) with w = D_rest / (D_obs + D_rest)."""
    total = D_obs + D_rest
    w = 0.0 if total == 0 else D_rest / total
    return (1.0 - w) * r_obs + w * r_rest, w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forecast.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/forecast.py tests/test_forecast.py
git commit -m "feat(forecast): final-line blend identity r_final=(1-w)r_obs+w·r_rest"
```

---

## Task 1: PA frame with `game_date` (`src/talent3.py`)

**Files:**
- Create: `src/talent3.py`
- Test: `tests/test_talent3.py`

**Context:** `talent2.build_pa_measurements` computes the per-PA value/denom but drops `game_date`. Cutpoints need temporal order, so `talent3` needs a PA frame that keeps `game_date`. The value logic is identical to Phase 1/Level 2 (BBE → `estimated_woba_using_speedangle` coalesce `woba_value`; else `woba_value`; drop rows with null `woba_denom`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_talent3.py
import numpy as np
import polars as pl
from src.talent3 import build_pa_frame


def _pitches():
    return pl.DataFrame({
        "batter":     [1, 1, 1, 2, 2, 1],
        "game_year":  [2024, 2024, 2024, 2024, 2024, 2024],
        "game_date":  ["2024-04-02", "2024-04-01", "2024-05-01",
                       "2024-04-01", "2024-04-02", "2024-04-03"],
        "type":       ["X", "X", "B", "X", "S", "S"],
        "estimated_woba_using_speedangle": [1.2, 0.1, None, 0.8, None, None],
        "woba_value": [1.25, 0.0, 0.69, 0.9, 0.0, 0.0],
        "woba_denom": [1, 1, 1, 1, 1, None],   # last row: non-PA, dropped
    })


def test_build_pa_frame_keeps_date_and_value_logic():
    f = build_pa_frame(_pitches())
    assert set(f.columns) == {"batter", "season", "game_date", "value", "denom"}
    assert f.height == 5                              # non-PA row dropped
    p1 = f.filter(pl.col("batter") == 1).sort("value")
    assert p1["value"].to_list() == [0.1, 0.69, 1.2]  # BBE→est_woba, walk→woba_value
    assert f["game_date"].dtype == pl.Date            # parsed for ordering
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_talent3.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/talent3.py  (module header + first function)
"""Rest-of-season xwOBA forecast: a hierarchical player-talent model evaluated at
mid-season cutpoints (spec: docs/superpowers/specs/2026-07-20-xwobart-rest-of-season-
forecast-design.md). Rung a: career random intercept + iid season drift, xwOBA-only,
no aging. Reuses talent2's bootstrap_S for the measurement variance. Pure functions;
orchestration in scripts/run_talent3.py."""
from __future__ import annotations

import numpy as np
import polars as pl

from src.talent2 import FLOOR_SD_PER_PA, bootstrap_S


def build_pa_frame(pitches: pl.DataFrame) -> pl.DataFrame:
    """One row per PA with (batter, season, game_date, value, denom). value/denom
    exactly as talent2.build_pa_measurements; game_date is parsed to pl.Date so
    PAs can be ordered within a season for cutpoints."""
    return (
        pitches.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value")),
            game_date=pl.col("game_date").cast(pl.Utf8).str.slice(0, 10)
            .str.to_date("%Y-%m-%d"),
        )
        .select("batter", season="game_year", game_date="game_date",
                value="value", denom="woba_denom")
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_talent3.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py tests/test_talent3.py
git commit -m "feat(talent3): PA frame carrying game_date for cutpoints"
```

---

## Task 2: Cutpoint split (`src/talent3.py`)

**Files:**
- Modify: `src/talent3.py`
- Test: `tests/test_talent3.py`

**Context:** For one player-season, order PAs by `game_date` (stable), take the first *k* as observed, the rest as remaining. Return the pieces the blend needs, plus the observed/remaining value arrays for downstream measurement and forward-bootstrap. Eligibility: remaining denom `≥ min_remaining` (else ineligible).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_talent3.py
from src.talent3 import cutpoint_split

def test_cutpoint_split_orders_and_blends():
    f = (build_pa_frame(_pitches()).filter(pl.col("batter") == 2))  # 2 PAs: .8, .0
    # k=1 → observed = earliest PA (2024-04-01 value .8), remaining = .0
    cp = cutpoint_split(f, k=1, min_remaining=1)
    assert cp is not None
    assert abs(cp["r_obs"] - 0.8) < 1e-12 and cp["D_obs"] == 1
    assert abs(cp["r_rest"] - 0.0) < 1e-12 and cp["D_rest"] == 1
    assert abs(cp["w"] - 0.5) < 1e-12
    # blend identity holds against the whole-season rate
    from src.forecast import final_line_blend
    r_final, _ = final_line_blend(cp["r_obs"], cp["D_obs"], cp["r_rest"], cp["D_rest"])
    assert abs(r_final - (0.8 + 0.0) / 2) < 1e-12

def test_cutpoint_split_ineligible_when_no_runway():
    f = build_pa_frame(_pitches()).filter(pl.col("batter") == 2)
    assert cutpoint_split(f, k=1, min_remaining=5) is None   # only 1 remaining PA
    assert cutpoint_split(f, k=2, min_remaining=1) is None   # nothing remaining
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_talent3.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# add to src/talent3.py
def cutpoint_split(pas: pl.DataFrame, k: int, min_remaining: float) -> dict | None:
    """Order one player-season's PAs by game_date, split first-k (observed) vs rest
    (remaining). Returns a dict with r_obs/D_obs, r_rest/D_rest (realized), w, and
    the raw observed/remaining value+denom arrays. None if fewer than k+1 PAs or the
    remaining denom < min_remaining (no real 'rest of season')."""
    o = pas.sort("game_date", maintain_order=True)   # stable: ties keep input order
    n = o.height
    if n <= k:
        return None
    v = o["value"].to_numpy(); d = o["denom"].to_numpy()
    v_obs, d_obs = v[:k], d[:k]
    v_rest, d_rest = v[k:], d[k:]
    D_obs, D_rest = float(d_obs.sum()), float(d_rest.sum())
    if D_rest < min_remaining or D_obs <= 0:
        return None
    return {
        "r_obs": float(v_obs.sum() / D_obs), "D_obs": D_obs,
        "r_rest": float(v_rest.sum() / D_rest), "D_rest": D_rest,
        "w": D_rest / (D_obs + D_rest),
        "v_obs": v_obs, "d_obs": d_obs, "v_rest": v_rest, "d_rest": d_rest,
    }
```

- [ ] **Step 4: Run to verify it passes** — Expected PASS.

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py tests/test_talent3.py
git commit -m "feat(talent3): cutpoint split with game_date ordering and eligibility"
```

---

## Task 3: Per-sample measurement — rate + `S[0,0]` via `bootstrap_S` (`src/talent3.py`)

**Files:**
- Modify: `src/talent3.py`
- Test: `tests/test_talent3.py`

**Context:** A sample (a season, or a first-*k* slice) yields a measurement `z = rate` and a sampling variance `s00`. Reuse `talent2.bootstrap_S` with NaN peripheral arrays and take `S[0,0]` (spec §4.2: the base rung consumes only `S[0,0]`; `bootstrap_S` still needs `ev`/`barrel` args and returns 3×3, but the `S[0,0]` branch at `talent2.py:100-102` runs independently). Variance floored at `FLOOR_SD_PER_PA² / n`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_talent3.py
from src.talent3 import sample_measurement

def test_sample_measurement_matches_bootstrap_S_diag():
    rng = np.random.default_rng(0)
    v = np.array([1.2, 0.1, 0.69, 0.0, 2.0, 0.0]); d = np.ones_like(v)
    z, s00 = sample_measurement(v, d, B=500, rng=rng)
    assert abs(z - v.sum() / d.sum()) < 1e-12
    # equals bootstrap_S[0,0] under the same seed
    from src.talent2 import bootstrap_S
    nan = np.full_like(v, np.nan)
    S = bootstrap_S(v, d, nan, nan, B=500, rng=np.random.default_rng(0))
    assert abs(s00 - S[0, 0]) < 1e-12

def test_sample_measurement_floor_binds_on_degenerate_sample():
    rng = np.random.default_rng(1)
    v = np.array([0.0, 0.0]); d = np.array([1.0, 1.0])   # zero variation
    _, s00 = sample_measurement(v, d, B=200, rng=rng)
    assert abs(s00 - FLOOR_SD_PER_PA ** 2 / 2) < 1e-12   # floored, not zero
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

```python
# add to src/talent3.py
def sample_measurement(value: np.ndarray, denom: np.ndarray, B: int,
                       rng: np.random.Generator) -> tuple[float, float]:
    """Measurement (rate) and its floored bootstrap sampling variance for one PA
    sample. xwOBA-only: reuse bootstrap_S with NaN peripherals and take S[0,0]."""
    nan = np.full(len(value), np.nan)
    S = bootstrap_S(value, denom, nan, nan, B=B, rng=rng)
    z = float(value.sum() / denom.sum())
    return z, float(S[0, 0])
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py tests/test_talent3.py
git commit -m "feat(talent3): per-sample measurement reusing bootstrap_S diagonal"
```

---

## Task 4: Causal per-season environment `μ_t` (`src/talent3.py`)

**Files:**
- Modify: `src/talent3.py`
- Test: `tests/test_talent3.py`

**Context (spec §4.1, §14):** `μ_t` centers the prior on the season's offensive environment and MUST be estimated causally. Chosen estimator: **league mean rate over all players' first-*k* PAs** in season *t* (order each player's PAs by `game_date`, take first *k*, pool). Leak-free (no PA beyond any cutpoint). Special case `k = ∞` (full season) → plain season league mean, used by G5.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_talent3.py
from src.talent3 import season_mu_causal

def test_season_mu_causal_uses_only_first_k():
    # Two players, 2024. With k=1, only each player's earliest PA counts.
    f = build_pa_frame(_pitches()).filter(pl.col("season") == 2024)
    # player1 earliest (04-01) value .1 ; player2 earliest (04-01) value .8
    mu_k1 = season_mu_causal(f, season=2024, k=1)
    assert abs(mu_k1 - (0.1 + 0.8) / 2) < 1e-12
    # full-season (k huge) = pooled league rate over all PAs
    mu_full = season_mu_causal(f, season=2024, k=10_000)
    all_v = f["value"].to_numpy(); all_d = f["denom"].to_numpy()
    assert abs(mu_full - all_v.sum() / all_d.sum()) < 1e-12
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

```python
# add to src/talent3.py
def season_mu_causal(pa_frame: pl.DataFrame, season: int, k: int) -> float:
    """League environment for `season`: pooled rate over every player's first-k PAs
    (ordered by game_date). Causal — uses no PA beyond a cutpoint. k huge → full season."""
    s = (
        pa_frame.filter(pl.col("season") == season)
        .sort("game_date", maintain_order=True)
        .with_columns(_rank=pl.col("game_date").cum_count().over("batter"))
        .filter(pl.col("_rank") <= k)
    )
    return float(s["value"].sum() / s["denom"].sum())
```

*Note:* `cum_count().over("batter")` after a global `game_date` sort ranks each player's PAs in date order; keep it grouped so partition order is deterministic. Verify with the test.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py tests/test_talent3.py
git commit -m "feat(talent3): causal per-season environment mu_t (all players' first-k)"
```

---

## Task 5: EB hyperparameter fit — `σ_η²`, `σ_u²` (`src/talent3.py`)

**Files:**
- Modify: `src/talent3.py`
- Test: `tests/test_talent3.py`

**Context (spec §4.3):** For player *i* with completed seasons *s*, the demeaned measurements `y_{i,s} = z_{i,s} − μ_s` are, marginalizing out `(η_i, {u_{i,s}})`,
`y_i ~ N(0, Σ_i)`, `Σ_i = σ_η² · 11ᵀ + diag(σ_u² + S_{i,s})`.
Fit `(σ_η², σ_u²)` by minimizing the summed marginal NLL over players (L-BFGS-B on log-params for positivity). Only players with ≥1 completed season contribute; a player with a single season contributes `σ_η²+σ_u²+S` on the diagonal (so `σ_η²` and `σ_u²` are jointly identified only from players with ≥2 seasons — expected with a 4-season window).

- [ ] **Step 1: Write the failing test** (synthetic recovery)

```python
# add to tests/test_talent3.py
from src.talent3 import fit_hypers_eb

def test_fit_hypers_recovers_known_variances():
    rng = np.random.default_rng(7)
    sig_eta, sig_u = 0.030, 0.015
    # 400 players, 3 seasons each, known measurement noise S≈0.02² per obs
    ys, Ss, pid = [], [], []
    for i in range(400):
        eta = rng.normal(0, sig_eta)
        for s in range(3):
            u = rng.normal(0, sig_u)
            S = 0.02 ** 2
            ys.append(eta + u + rng.normal(0, np.sqrt(S)))
            Ss.append(S); pid.append(i)
    est_eta2, est_u2 = fit_hypers_eb(np.array(ys), np.array(Ss), np.array(pid))
    assert abs(np.sqrt(est_eta2) - sig_eta) < 0.006   # within ~1 sampling SE
    assert abs(np.sqrt(est_u2) - sig_u) < 0.006
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

```python
# add to src/talent3.py
from scipy.optimize import minimize

def fit_hypers_eb(y: np.ndarray, S: np.ndarray, player_idx: np.ndarray
                  ) -> tuple[float, float]:
    """Marginal-MLE of (sigma_eta^2, sigma_u^2) for the rung-a hierarchy.
    y = z - mu_s (demeaned measurements), S = per-obs measurement variance,
    player_idx groups rows by player. Per player: y_i ~ N(0, sig_eta^2 11' +
    diag(sig_u^2 + S_i)). Optimizes log-variances (positivity) with L-BFGS-B."""
    groups = [np.where(player_idx == p)[0] for p in np.unique(player_idx)]

    def nll(theta: np.ndarray) -> float:
        se2, su2 = np.exp(theta)
        total = 0.0
        for g in groups:
            yi, Si = y[g], S[g]
            Sig = se2 * np.ones((len(g), len(g))) + np.diag(su2 + Si)
            sign, logdet = np.linalg.slogdet(Sig)
            sol = np.linalg.solve(Sig, yi)
            total += 0.5 * (logdet + yi @ sol)
        return float(total)

    res = minimize(nll, np.log([0.02 ** 2, 0.01 ** 2]),
                   method="L-BFGS-B", options={"maxiter": 200})
    se2, su2 = np.exp(res.x)
    return float(se2), float(su2)
```

*Performance note:* ~2,600 player-seasons across ~1,000 players, groups ≤4 rows → the loop is fast; no need to vectorize for the fit. If it is ever a bottleneck, batch equal-size groups.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py tests/test_talent3.py
git commit -m "feat(talent3): EB marginal-MLE of career/drift variances"
```

---

## Task 6: Closed-form per-cutpoint posterior (`src/talent3.py`)

**Files:**
- Modify: `src/talent3.py`
- Test: `tests/test_talent3.py`

**Context (spec §4.4):** Given `(σ_η², σ_u²)`, per-season `μ_s`, a player's completed prior-season measurements `{(z_s, S_s): s < t}`, and the first-*k* current measurement `(z_t, S_t)`, the posterior of `θ_{i,t} = μ_t + η_i + u_{i,t}` is closed-form. Latent vector `x = (η, u_{s1}, …, u_{sJ}, u_t)`; prior `x ~ N(0, blkdiag(σ_η², σ_u² I))`; each measurement `z_s = μ_s + η + u_s + ε_s`, so demeaned `y_s = z_s − μ_s = H_s x + ε_s` with `H_s` picking `(η, u_s)`. Standard Gaussian linear update → posterior mean/cov of `x`; then `θ̂ = μ_t + ĥ_η + ĥ_{u_t}`, `V = Var(η + u_t)`.

- [ ] **Step 1: Write the failing test** (match brute-force + G5 reduction)

```python
# add to tests/test_talent3.py
from src.talent3 import cutpoint_posterior

def test_posterior_matches_brute_force_gaussian():
    se2, su2 = 0.030 ** 2, 0.015 ** 2
    # prior seasons: two measurements; current: one truncated measurement
    zs = np.array([0.360, 0.330, 0.345]); mus = np.array([0.315, 0.318, 0.320])
    Ss = np.array([0.020 ** 2, 0.022 ** 2, 0.045 ** 2])   # current (last) is noisier
    is_current = np.array([False, False, True])
    theta, V = cutpoint_posterior(zs, mus, Ss, is_current, se2, su2)

    # brute force: latent x=(eta,u0,u1,u_t); build joint, condition on y=z-mu
    P = np.diag([se2, su2, su2, su2])
    H = np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1]], float)
    R = np.diag(Ss); y = zs - mus
    Kf = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    xhat = Kf @ y; Vx = P - Kf @ H @ P
    sel = np.array([1,0,0,1], float)   # eta + u_t
    assert abs(theta - (mus[-1] + sel @ xhat)) < 1e-10
    assert abs(V - sel @ Vx @ sel) < 1e-10

def test_posterior_no_history_reduces_to_1d_shrinkage():
    # single current measurement, no prior seasons → Phase-1/L2 1-D shrink toward mu_t
    se2, su2 = 0.030 ** 2, 0.015 ** 2
    z, mu, S = np.array([0.400]), np.array([0.315]), np.array([0.050 ** 2])
    theta, V = cutpoint_posterior(z, mu, S, np.array([True]), se2, su2)
    tau2 = se2 + su2                       # prior var of (eta+u_t) with no history
    rel = tau2 / (tau2 + S[0])
    assert abs(theta - (mu[0] + rel * (z[0] - mu[0]))) < 1e-10
    assert abs(V - rel * S[0]) < 1e-10
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** (general Gaussian linear update)

```python
# add to src/talent3.py
def cutpoint_posterior(z: np.ndarray, mu: np.ndarray, S: np.ndarray,
                       is_current: np.ndarray, se2: float, su2: float
                       ) -> tuple[float, float]:
    """Posterior (theta_hat, V) of theta_{i,t}=mu_t+eta+u_t given demeaned
    measurements y=z-mu with per-obs variance S. Latent x=(eta, u_1..u_J); the
    current season's u is the last component. Closed-form Gaussian conditioning."""
    J = len(z)
    P = np.diag(np.concatenate([[se2], np.full(J, su2)]))      # (J+1, J+1)
    H = np.zeros((J, J + 1))
    H[:, 0] = 1.0                                              # eta loads on all
    H[np.arange(J), 1 + np.arange(J)] = 1.0                    # each obs -> its own u
    R = np.diag(S)
    y = z - mu
    Kf = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    xhat = Kf @ y
    Vx = P - Kf @ H @ P
    cur = int(np.where(is_current)[0][0])                      # index of current season
    sel = np.zeros(J + 1); sel[0] = 1.0; sel[1 + cur] = 1.0    # eta + u_current
    theta = float(mu[cur] + sel @ xhat)
    V = float(sel @ Vx @ sel)
    return theta, max(V, 0.0)
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py tests/test_talent3.py
git commit -m "feat(talent3): closed-form per-cutpoint posterior (Gaussian linear update)"
```

---

## Task 7: Forward-bootstrap forecaster (`src/forecast.py`)

**Files:**
- Modify: `src/forecast.py`
- Test: `tests/test_forecast.py`

**Context (spec §5):** Combine estimation (`θ̂, V`) with future-sample noise over `m` remaining PAs and blend into the final line. For `b = 1..B`: draw `θ_b ~ N(θ̂, V)`; resample `m` (value, denom) pairs from a reference multiset, form a rate, **additive-shift** it so its mean is `θ_b` (preserves the boom/bust asymmetry); `final_b = (1−w)·r_obs + w·r_rest_b`. Report center + 50/80/90 quantiles. `m` = number of remaining PAs (`len(ref)`-independent; use realized remaining count for validation).

- [ ] **Step 1: Write the failing test** (coverage + collapse + asymmetry)

```python
# add to tests/test_forecast.py
from src.forecast import forward_forecast

def test_forward_forecast_nominal_coverage_on_synthetic():
    rng = np.random.default_rng(3)
    # true talent theta*, symmetric value dist; check the 80% interval covers ~80%
    theta_star, V = 0.330, 0.010 ** 2
    ref_v = rng.normal(0.330, 0.30, size=2000); ref_d = np.ones_like(ref_v)
    hits = 0; T = 400; m = 150
    for _ in range(T):
        # realized rest-of-season rate at the true talent
        idx = rng.integers(0, len(ref_v), size=m)
        r_actual = ref_v[idx].mean() + (theta_star - ref_v.mean())
        out = forward_forecast(theta_star, V, r_obs=0.330, w=1.0,
                               ref_v=ref_v, ref_d=ref_d, m=m, B=600, rng=rng)
        lo, hi = out["q10"], out["q90"]          # 80% central interval
        hits += lo <= r_actual <= hi
    assert 0.72 <= hits / T <= 0.88               # ~80% ± tolerance

def test_forward_forecast_collapses_when_no_runway():
    rng = np.random.default_rng(4)
    ref_v = rng.normal(0.33, 0.3, 500); ref_d = np.ones_like(ref_v)
    out = forward_forecast(0.33, 0.02 ** 2, r_obs=0.351, w=0.0,
                           ref_v=ref_v, ref_d=ref_d, m=0, B=300, rng=rng)
    assert abs(out["center"] - 0.351) < 1e-9 and (out["q90"] - out["q10"]) < 1e-9

def test_forward_forecast_preserves_right_skew():
    rng = np.random.default_rng(5)
    ref_v = rng.exponential(0.3, 4000); ref_d = np.ones_like(ref_v)   # right-skewed
    out = forward_forecast(0.30, 0.005 ** 2, r_obs=0.30, w=1.0,
                           ref_v=ref_v, ref_d=ref_d, m=120, B=4000, rng=rng)
    assert (out["q90"] - out["center"]) > (out["center"] - out["q10"])  # upside room
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

```python
# add to src/forecast.py
def forward_forecast(theta_hat: float, V: float, r_obs: float, w: float,
                     ref_v: np.ndarray, ref_d: np.ndarray, m: int, B: int,
                     rng: np.random.Generator,
                     levels=(0.5, 0.8, 0.9)) -> dict:
    """Final-line predictive summary. Draw theta ~ N(theta_hat, V); forward-bootstrap
    an m-PA rest-of-season rate from (ref_v, ref_d) additively shifted to mean theta;
    blend by w. Returns center and lo/hi at each level (keys q<pct>)."""
    if m <= 0 or w == 0.0:
        base = (1.0 - w) * r_obs + w * theta_hat
        return {"center": base, **{k: base for k in _level_keys(levels)}}
    thetas = rng.normal(theta_hat, np.sqrt(max(V, 0.0)), size=B)
    ref_mean = ref_v.sum() / ref_d.sum()
    finals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(ref_v), size=m)
        rate = ref_v[idx].sum() / ref_d[idx].sum()
        r_rest = rate + (thetas[b] - ref_mean)         # additive shift -> mean theta_b
        finals[b] = (1.0 - w) * r_obs + w * r_rest
    out = {"center": float(np.median(finals))}
    for lv in levels:
        lo, hi = (1 - lv) / 2, 1 - (1 - lv) / 2
        out[f"q{int(lo*100):02d}"] = float(np.quantile(finals, lo))
        out[f"q{int(hi*100):02d}"] = float(np.quantile(finals, hi))
    return out


def _level_keys(levels):
    keys = []
    for lv in levels:
        lo = (1 - lv) / 2
        keys += [f"q{int(lo*100):02d}", f"q{int((1-lo)*100):02d}"]
    return keys
```

*Note:* levels (.5,.8,.9) → keys q25/q75, q10/q90, q05/q95. The test uses q10/q90 (80%). Confirm key names match before writing the metrics task.

- [ ] **Step 4: Run to verify it passes.** (Coverage test is stochastic; if it flakes at the boundary, widen tolerance to [0.70, 0.90] — the point is *approximate* nominal coverage, not a tight estimate.)

- [ ] **Step 5: Commit**

```bash
git add src/forecast.py tests/test_forecast.py
git commit -m "feat(forecast): forward-bootstrap final-line predictive interval"
```

---

## Task 8: Benchmarks (`src/benchmarks.py`)

**Files:**
- Create: `src/benchmarks.py`
- Test: `tests/test_benchmarks.py`

**Context (spec §8):** Five point forecasts of the rest-of-season rate (→ final line via the blend). `naive` = `r_obs`. `league_shrunk` = Phase-1 EB shrink of the first-*k* rate toward `μ_t` (reuse `src/talent.eb_shrink` semantics). `single_season_l2` = the Level-2 estimate on the first-*k* sample (reuse `talent2`; wired in the script). `marcel` = weighted blend of prior-season rates (weights 5/4/3 by recency) + current-to-date, regressed toward `μ_t` by adding `regress_pa` league PAs. `savant_to_date` = Savant's season xwOBA through *k* (from the data; wired in the script). Implement the pure ones here; `single_season_l2`/`savant` are assembled in the script from existing tables.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py
import numpy as np
from src.benchmarks import naive, marcel

def test_naive_is_observed_rate():
    assert naive(0.351) == 0.351

def test_marcel_weights_and_regression():
    # one prior season rate .340 over 400 denom; current .360 over 100 denom;
    # league mu .315; regress with 200 league PA. Weighted-mean then shrink.
    est = marcel(prior_rates=[0.340], prior_denoms=[400.0],
                 cur_rate=0.360, cur_denom=100.0, mu=0.315,
                 regress_pa=200.0, weights=(5, 4, 3))
    # manual: recency weight prior=5*400, current gets weight 5*100? -> see impl doc
    assert 0.315 < est < 0.360           # between league and current, sensible
    # regression pulls toward mu: with regress_pa large, est -> mu
    est_heavy = marcel([0.340], [400.0], 0.360, 100.0, 0.315,
                       regress_pa=100_000.0)
    assert abs(est_heavy - 0.315) < 0.01
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

```python
# src/benchmarks.py
"""Benchmark rest-of-season-rate forecasts for the talent3 evaluation (spec §8).
Pure functions; single_season_l2 and savant_to_date are assembled in the script
from existing tables."""
from __future__ import annotations


def naive(r_obs: float) -> float:
    """'He keeps hitting as he has.'"""
    return r_obs


def marcel(prior_rates, prior_denoms, cur_rate: float, cur_denom: float,
           mu: float, regress_pa: float = 200.0, weights=(5, 4, 3)) -> float:
    """Marcel-style projection: recency-weighted mean of {prior seasons (most recent
    first), current-to-date}, then regressed toward the league mean mu by adding
    regress_pa league-average PAs. Denominator-weighted within each season."""
    num = cur_rate * cur_denom * weights[0]
    den = cur_denom * weights[0]
    for j, (r, d) in enumerate(zip(prior_rates, prior_denoms)):
        w = weights[min(j + 1, len(weights) - 1)]
        num += r * d * w
        den += d * w
    num += mu * regress_pa
    den += regress_pa
    return num / den
```

*Doc for the test:* current season gets the top weight (5); prior seasons take 4, 3, … by recency. Regression adds `regress_pa` PAs at the league mean. Confirm the test's bounds hold; adjust the weight assignment only if the design review of this file flags it.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks.py tests/test_benchmarks.py
git commit -m "feat(benchmarks): naive and Marcel rest-of-season baselines"
```

---

## Task 9: Validation sweep + leakage guard (`scripts/run_talent3.py`)

**Files:**
- Create: `scripts/run_talent3.py`
- Test: `tests/test_talent3.py` (leakage-guard unit test on a pure helper)

**Context:** Orchestrate the whole pipeline. Load pitches (`run_talent.load_pitches`), build the PA frame, and for each season *t* fit `φ` **leave-one-season-out** (on the other three full seasons' player measurements), compute `μ_s` causally, then sweep every eligible (player, *k*): build prior-season measurements + first-*k* measurement, `cutpoint_posterior`, `forward_forecast`, and all benchmarks. Write `results/talent3/forecast_table.parquet`. A pure helper `assert_causal(rows, cutpoint_date, season_t)` raises if any conditioning row has `game_date > cutpoint_date` or `season > season_t`; call it per forecast and stamp a digest in metrics.

- [ ] **Step 1: Write the failing test** (leakage guard is a pure, unit-testable function)

```python
# add to tests/test_talent3.py
from src.talent3 import assert_causal
import datetime as dt

def test_assert_causal_flags_future_rows():
    cut = dt.date(2024, 6, 1)
    ok = pl.DataFrame({"game_date":[dt.date(2024,5,1)], "season":[2024]})
    assert_causal(ok, cut, 2024)                     # no raise
    future = pl.DataFrame({"game_date":[dt.date(2024,7,1)], "season":[2024]})
    try:
        assert_causal(future, cut, 2024); raised = False
    except AssertionError:
        raised = True
    assert raised
    later_season = pl.DataFrame({"game_date":[dt.date(2023,5,1)], "season":[2025]})
    try:
        assert_causal(later_season, cut, 2024); raised = False
    except AssertionError:
        raised = True
    assert raised
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `assert_causal` in `src/talent3.py`, then the orchestration in `scripts/run_talent3.py`.

```python
# add to src/talent3.py
def assert_causal(rows: pl.DataFrame, cutpoint_date, season_t: int) -> None:
    """Raise if any conditioning row leaks the future: a PA after the cutpoint date
    or a season strictly after the target season."""
    if rows.height == 0:
        return
    assert rows.filter(pl.col("game_date") > cutpoint_date).height == 0, \
        "leak: conditioning PA after cutpoint date"
    assert rows.filter(pl.col("season") > season_t).height == 0, \
        "leak: conditioning row from a later season"
```

Then `scripts/run_talent3.py` (structure — mirror `run_talent2.py`; full body written during implementation):

```python
# scripts/run_talent3.py  (skeleton — fill in during the task)
#   CUTPOINTS = [50, 100, 150, 200, 300]; MIN_REMAINING = 30; B_BOOT = 500; B_FWD = 600
#   cfg = load_config(); pit = load_pitches(cfg, cfg.all_seasons); f = build_pa_frame(pit)
#   for t in cfg.all_seasons:
#       phi = fit_hypers_eb(... LOSO: players' full-season measurements for seasons != t ...)
#       for (player, k) in eligible cutpoints in season t:
#           cp = cutpoint_split(player_season_pas, k, MIN_REMAINING)
#           # measurements: prior full seasons (< t) + first-k of t ; assert_causal(...)
#           mu = [season_mu_causal(f, s, k_for_s) for each season used]
#           theta, V = cutpoint_posterior(z, mu, S, is_current, *phi)
#           fc = forward_forecast(theta, V, cp["r_obs"], cp["w"], ref_v, ref_d,
#                                 m=len(cp["v_rest"]), B=B_FWD, rng=rng)
#           r_final_actual, _ = final_line_blend(cp["r_obs"], cp["D_obs"],
#                                                cp["r_rest"], cp["D_rest"])
#           bench = {naive, league_shrunk, single_season_l2, marcel, savant_to_date}
#           append row(batter, season, k, w, theta, fc quantiles, r_final_actual, bench...)
#   write results/talent3/forecast_table.parquet ; stamp leakage digest in metrics
```

- [ ] **Step 4: Run to verify** — `.venv/bin/python -m pytest tests/test_talent3.py -v` (unit) then `.venv/bin/python scripts/run_talent3.py` produces `results/talent3/forecast_table.parquet`. Print row count and eligible-pair counts per (season, k) band.

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py scripts/run_talent3.py tests/test_talent3.py
git commit -m "feat(talent3): validation sweep with LOSO hypers and leakage guard"
```

---

## Task 10: Gate scoring + metrics (`scripts/run_talent3.py`)

**Files:**
- Modify: `scripts/run_talent3.py`
- Test: `tests/test_talent3.py`

**Context (spec §9):** From `forecast_table`, compute point RMSE of `r_final` vs actual for the model and each benchmark, by **PA-seen** (*k*) and **w** band, pooled, with a paired bootstrap over players (reuse `run_talent2.paired_bootstrap` pattern). Coverage of the 50/80/90 intervals by band (share of actuals within `[q_lo, q_hi]`, target within ±5pp). Gates:
- **G1** model beats naive on final-line RMSE at low PA-seen (k≤100), paired-bootstrap CI excludes 0.
- **G2** model beats/ties single-season L2.
- **G3** model beats/ties Marcel.
- **G4** coverage within ±5pp at 50/80/90 across bands.
- **G5** with history stripped (η removed) at the full-season cutpoint and `μ_t`=full-season league mean, estimates match `results/talent2` to tolerance.

- [ ] **Step 1: Write the failing test** (a pure scoring helper)

```python
# add to tests/test_talent3.py
from src.talent3 import coverage_by_band

def test_coverage_by_band_counts_hits():
    tbl = pl.DataFrame({
        "band": ["50","50","100","100"],
        "actual": [0.30, 0.40, 0.31, 0.36],
        "q05": [0.25, 0.25, 0.28, 0.30], "q95": [0.35, 0.35, 0.34, 0.40],
    })
    cov = coverage_by_band(tbl, "band", 0.90)     # share within [q05,q95] per band
    assert cov["50"] == 0.5      # .30 in, .40 out
    assert cov["100"] == 1.0     # both in
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `coverage_by_band` in `src/talent3.py` and the gate/metrics block in `scripts/run_talent3.py`; write `results/talent3/metrics.json` (gates, by-band RMSE tables, coverage tables, paired bootstraps). Reuse `_gate`, `_calibrated_rmse`, `_pearson`, `paired_bootstrap`.

- [ ] **Step 4: Run to verify** — unit test passes; `scripts/run_talent3.py` prints each GATE line and writes `metrics.json`. **Record actual G1–G5 outcomes honestly** (a documented non-beat is a valid result; do not tune to rescue a gate — same discipline as Stage 3).

- [ ] **Step 5: Commit**

```bash
git add src/talent3.py scripts/run_talent3.py tests/test_talent3.py
git commit -m "feat(talent3): gate scoring, by-band RMSE + coverage, paired bootstrap"
```

---

## Task 11: Figures, NOTES, RESULTS (`scripts/run_talent3.py`, docs)

**Files:**
- Modify: `scripts/run_talent3.py`
- Create: `results/talent3/NOTES.md`
- Modify: `results/RESULTS.md`

**Context:** Produce the product's headline visuals and the honest write-up. Follow the existing NOTES voice (Level-2 NOTES): lead with the mechanism, report gates with their CIs, name limitations plainly (affine-invariance caveat → report pooled RMSE + by-band; do not quote a gain without its paired-bootstrap CI).

- [ ] **Step 1: Figures** — add to `run_talent3.py`, saved under `results/talent3/figures/`:
  - `fan_chart_examples.png` — 3–4 example players: xwOBA-to-date with the final-line forecast fan (50/80/90) narrowing across cutpoints toward the realized final line. **The product shot.**
  - `calibration_by_band.png` — empirical vs nominal coverage at 50/80/90, faceted by PA-seen and by *w*.
  - `width_vs_pa_and_w.png` — mean interval width vs *k* and vs *w*.
  - `rmse_vs_benchmarks.png` — final-line RMSE by band, model vs the five benchmarks.

- [ ] **Step 2: Run** `.venv/bin/python scripts/run_talent3.py` — confirm all figures render and `metrics.json` is current.

- [ ] **Step 3: Write `results/talent3/NOTES.md`** — goal, construction (rung a), the gate panel with CIs, the fan chart callout, and limitations (§12: 4-season window, no aging/peripherals yet, thin forward-bootstrap at low k, Gaussian tails, selection at high k). State which gates passed and which did not, with numbers.

- [ ] **Step 4: Add a `## Rest-of-season xwOBA forecast (rung a)` section to `results/RESULTS.md`** — headline gate outcomes, the by-band RMSE-vs-benchmark table, the coverage table, and a one-line pointer to `results/talent3/NOTES.md`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_talent3.py results/talent3/NOTES.md results/RESULTS.md
git commit -m "docs(talent3): fan chart + calibration figures, NOTES, RESULTS section"
```

---

## Verification checklist (before declaring done)

- [ ] `.venv/bin/python -m pytest tests/test_talent3.py tests/test_forecast.py tests/test_benchmarks.py -v` — all green. Use @superpowers-extended-cc:verification-before-completion: paste the actual pass/fail counts; do not claim green without the output.
- [ ] `scripts/run_talent3.py` runs end-to-end and writes `forecast_table.parquet`, `metrics.json`, and all figures.
- [ ] **G5 holds** (reduce-to-Level-2 at full-season cutpoint) — this is the load-bearing correctness check that the hierarchy machinery is right.
- [ ] Leakage digest present in `metrics.json`; `assert_causal` wired into the sweep.
- [ ] NOTES + RESULTS report gate outcomes **honestly**, with paired-bootstrap CIs; no gain quoted without its CI; pooled RMSE + by-band (never per-band r alone).
- [ ] Follow-on scope (rungs b/c) recorded in NOTES as the next levers.
