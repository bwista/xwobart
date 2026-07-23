# Rung b — spray peripheral for the in-season forecast — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking. Every code task uses @superpowers-extended-cc:test-driven-development.

**Goal:** Add a hitter's pull tendency as an early-settling peripheral in the rest-of-season forecaster so short-runway (low-PA) forecasts sharpen — the in-season regime where rung a is weakest.

**Architecture:** Change *only* the per-season talent measurement. Each season's scalar `(z, S)` that feeds the unchanged rung-a hierarchy (`cutpoint_posterior`) is replaced by a peripheral-informed `(z', S')` from a conditional-prior marginal likelihood (spec §4.2): θ stays flat in the measurement layer, peripherals borrow through the *talent* cross-covariance `Σ_θp`, and the single shrinkage happens once, in the hierarchy. Everything downstream (forward bootstrap, benchmarks, gate scoring) is untouched. A three-model ablation (xwOBA-only / +EV+barrel / +EV+barrel+pull) isolates spray's marginal value.

**Tech Stack:** Python 3.12, numpy, polars, scipy.optimize; pytest. Spec: `docs/superpowers/specs/2026-07-23-xwobart-forecast-rungb-spray-peripheral-design.md`.

**Reference before starting:** the spec (above); `src/talent2.py` (MVN machinery — `bootstrap_S`, `mvn_mle`, `mvn_posterior`); `src/talent3.py` (`build_pa_frame`, `sample_measurement`, `cutpoint_posterior`); `scripts/run_talent3.py` (`precompute_full_measurements`, `build_priors`, `forecast_row`, `run_sweep`, `run_scoring`, `main`); `src/prep.py:_spray_cols` (the signed pull angle). Run tests with `.venv/bin/python -m pytest`.

**Decision locked for the plan:** pull statistic = **mean `spray_pull` over tracked BBE** (continuous; symmetric treatment with `avg_ev`). If Task 3's pre-check prefers pull-rate, swap the aggregation only (the machinery is identical).

**Measurement-noise:** base rung takes `S` **block-diagonal** between xwOBA and the peripherals (`S[0,1:]=0`), per spec §4.2 — so the borrow is purely the talent correlation and the reduction identity is exact.

---

### Task 1: Carry EV / barrel / pull per PA in the forecaster's PA frame

**Files:**
- Modify: `src/talent3.py` — `build_pa_frame` (currently lines 17-32)
- Test: `tests/test_talent3.py`

Rung a's `build_pa_frame` keeps only `(batter, season, game_date, value, denom)`. Rung b needs `ev`, `barrel`, `pull` per PA (BBE-only, NaN otherwise), ordered by `game_date` for cutpoints. `pull` is **`spray_obs`** — the signed, handedness-mirrored pull angle that `src/prep.py:_spray_cols` adds (needs `hc_x/hc_y/stand`, already in `src/data.py:KEEP_COLUMNS`). **Use `spray_obs`, NOT `spray_pull`:** `spray_pull` is the *imputed* surface feature built later by `add_spray` (which needs lookup tables we don't want here); `spray_obs` is the raw observed angle, null on the ~0.04% of tracked BBE with missing `hc_x/hc_y` — fine, since `bootstrap_S` is NaN-aware. `ev`/`barrel` mirror `talent2.build_pa_measurements` exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_talent3.py
import numpy as np, polars as pl
from src.talent3 import build_pa_frame

def test_build_pa_frame_carries_peripherals():
    # two BBE (type X) + one non-BBE (walk). pull comes from _spray_cols(hc_x,hc_y,stand).
    pitches = pl.DataFrame({
        "batter": [1, 1, 1], "game_year": [2024, 2024, 2024],
        "game_date": ["2024-04-01", "2024-04-02", "2024-04-03"],
        "type": ["X", "X", None], "events": ["single", "home_run", "walk"],
        "description": ["hit_into_play", "hit_into_play", "walk"],
        "launch_speed": [95.0, 104.0, None],
        "launch_speed_angle": [4, 6, None],       # barrel = (==6)
        "estimated_woba_using_speedangle": [0.9, 2.0, None],
        "woba_value": [0.9, 2.0, 0.7], "woba_denom": [1, 1, 1],
        "hc_x": [100.0, 130.0, None], "hc_y": [100.0, 90.0, None],
        "stand": ["R", "R", "R"],
    })
    out = build_pa_frame(pitches).sort("game_date")
    assert out.columns[:5] == ["batter", "season", "game_date", "value", "denom"]
    assert set(["ev", "barrel", "pull"]).issubset(out.columns)
    ev = out["ev"].to_list()
    assert ev[0] == 95.0 and ev[1] == 104.0 and ev[2] is None        # non-BBE -> null
    assert out["barrel"].to_list() == [0.0, 1.0, None]
    pull = out["pull"].to_list()
    assert pull[0] is not None and pull[1] is not None and pull[2] is None
    # _spray_cols sign: hc_x=100 is left-of-home (pulled,+) for RHB, hc_x=130 is oppo (−);
    # test only checks non-null, so sign convention isn't asserted here
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_talent3.py::test_build_pa_frame_carries_peripherals -v` → FAIL (no `ev`/`barrel`/`pull` columns).

- [ ] **Step 3: Implement** — extend `build_pa_frame`. Apply `_spray_cols` to get `spray_pull`, then compute per-PA `ev/barrel/pull` on tracked BBE (reuse talent2's `tracked` predicate):

```python
from src.prep import _spray_cols   # top of file

def build_pa_frame(pitches: pl.DataFrame) -> pl.DataFrame:
    tracked = ((pl.col("type") == "X") & pl.col("launch_speed").is_not_null()
               & pl.col("launch_speed_angle").is_not_null())
    p = _spray_cols(pitches)        # adds spray_obs (signed pull angle, handedness-mirrored)
    return (
        p.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value")),
            game_date=pl.col("game_date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d"),
            ev=pl.when(tracked).then(pl.col("launch_speed")),
            barrel=pl.when(tracked).then((pl.col("launch_speed_angle") == 6).cast(pl.Float64)),
            pull=pl.when(tracked).then(pl.col("spray_obs")),
        )
        .select("batter", season="game_year", game_date="game_date",
                value="value", denom="woba_denom", ev="ev", barrel="barrel", pull="pull")
    )
```

Check `_spray_cols` input assumptions (it asserts `stand` non-null); if the forecast cache can carry null `stand` on non-BBE, guard by imputing/masking before `_spray_cols` and confirm the assert still holds on real data in Task 10.

- [ ] **Step 4: Run to verify it passes** — same command → PASS.
- [ ] **Step 5: Regression** — `.venv/bin/python -m pytest tests/test_talent3.py -v` → all prior rung-a tests still green (extra columns are additive; downstream selects by name).
- [ ] **Step 6: Commit** — `git add src/talent3.py tests/test_talent3.py && git commit -m "feat(talent3): carry ev/barrel/pull per PA for rung b"`

---

### Task 2: Generalize `bootstrap_S` to include the pull channel (k-channel, 3-channel bit-identical)

**Files:**
- Modify: `src/talent2.py` — `bootstrap_S` (lines 81-111)
- Test: `tests/test_talent2.py`

`bootstrap_S` hardcodes a 3×3 over `(xwoba, ev, barrel)`. Add `pull` as a 4th channel via an optional array arg so existing 3-channel callers are untouched and bit-identical. Pull replicates use the same BBE-valid mask logic as ev/barrel.

- [ ] **Step 1: Write failing tests** (two: back-compat + 4-channel shape/PSD)

```python
# tests/test_talent2.py
import numpy as np
from src.talent2 import bootstrap_S

def test_bootstrap_S_3channel_unchanged():
    # BIT-IDENTITY guard: before editing bootstrap_S, run this fixture on the CURRENT
    # function, print S, and paste it as S_baseline; then assert exact reproduction.
    v = np.array([0.5, 1.2, 0.0, 2.0, 0.3]); d = np.ones(5)
    ev = np.array([90., 101., 88., 104., np.nan]); br = np.array([0., 1., 0., 1., np.nan])
    S = bootstrap_S(v, d, ev, br, B=400, rng=np.random.default_rng(0))
    S_baseline = np.array([[...], [...], [...]])   # paste pre-change value (Step 0)
    assert S.shape == (3, 3)
    assert np.array_equal(S, S_baseline)           # bit-identical, not just finite

def test_bootstrap_S_4channel_pull():
    v = np.array([0.5, 1.2, 0.0, 2.0, 0.3]); d = np.ones(5)
    ev = np.array([90., 101., 88., 104., np.nan]); br = np.array([0., 1., 0., 1., np.nan])
    pull = np.array([-5., 18., 2., 22., np.nan])
    S = bootstrap_S(v, d, ev, br, B=400, rng=np.random.default_rng(0), pull=pull)
    assert S.shape == (4, 4)
    w = np.linalg.eigvalsh(S[np.ix_([0,1,2,3],[0,1,2,3])])
    assert (w >= -1e-9).all()          # PSD
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_talent2.py -k bootstrap_S -v` → the 4-channel test errors (no `pull` kwarg).
- [ ] **Step 3: Implement** — add `pull: np.ndarray | None = None`; when given, append a `pull_rep` (same NaN-aware BBE mask as `ev_rep`/`br_rep`) and build the covariance over the present channels; floor the pull diagonal like the other peripherals. Keep the exact 3-channel path when `pull is None` (compute over `[xw, ev_rep, br_rep]` as today). Confirm bit-identity by asserting `test_bootstrap_S_3channel_unchanged` value against a captured baseline.
- [ ] **Step 4: Run to verify pass** — same command → PASS (both).
- [ ] **Step 5: Regression** — `.venv/bin/python -m pytest tests/test_talent2.py -v` → all green (talent2 production path passes `pull=None`).
- [ ] **Step 6: Commit** — `git commit -am "feat(talent2): optional pull channel in bootstrap_S (3-channel unchanged)"`

---

### Task 3: The de-risk pre-check — does early pull predict rest-of-season xwOBA beyond EV/barrel? (§5 go/no-go)

**Files:**
- Create: `src/precheck.py` — pure function `pull_incremental_signal(...)`
- Modify: `scripts/run_talent3.py` — a `--precheck` branch that assembles inputs and prints the verdict
- Test: `tests/test_precheck.py`

Before building the model, confirm pull carries incremental signal. For each eligible `(batter, season, k)`, gather early-*k* `(xwoba, avg_ev, barrel, mean_pull)` and the realized rest-of-season xwOBA rate; per k-band, compare OLS incremental R² (or the partial correlation of pull) of adding pull to `{xwoba, ev, barrel}`. Cheap; no model fit.

- [ ] **Step 1: Write failing test** — synthetic where rest-rate depends on pull beyond ev/barrel ⇒ positive incremental R²; and a null construction ⇒ ~0.

```python
# tests/test_precheck.py
import numpy as np
from src.precheck import pull_incremental_signal

def test_pull_incremental_signal_detects_and_nulls():
    rng = np.random.default_rng(0); n = 4000
    xwoba = rng.normal(.32, .03, n); ev = rng.normal(89, 3, n); barrel = rng.uniform(0, .2, n)
    pull = rng.normal(8, 5, n)
    rest_signal = .3*xwoba + .002*pull + rng.normal(0, .02, n)   # pull matters
    rest_null   = .3*xwoba + .01*barrel + rng.normal(0, .02, n)  # pull irrelevant
    X = {"xwoba": xwoba, "ev": ev, "barrel": barrel, "pull": pull}
    assert pull_incremental_signal(X, rest_signal)["delta_r2"] > 0.01
    assert pull_incremental_signal(X, rest_null)["delta_r2"] < 0.005
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_precheck.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `pull_incremental_signal(X, y) -> {r2_base, r2_full, delta_r2, pull_partial_corr, n}` (two `np.linalg.lstsq` fits, base = `[1,xwoba,ev,barrel]`, full = base+`pull`). Report by band in the script layer.
- [ ] **Step 4: Run to verify pass** — PASS.
- [ ] **Step 5: Wire `--precheck`** in `run_talent3.py`: build the PA frame (Task 1), assemble per-(batter,season,k) early features + realized rest rate over `CUTPOINTS`, call the function per k-band, print a table and a GO/STOP verdict (GO if `delta_r2` non-trivial at low k). Write `results/talent3/precheck_pull.json`.
- [ ] **Step 6: Commit** — `git add src/precheck.py tests/test_precheck.py scripts/run_talent3.py && git commit -m "feat(talent3): pull incremental-signal pre-check (§5 go/no-go)"`

> **GATE:** run `--precheck` on real data now (Task 10 harness not required — it only needs Task 1). If `delta_r2` is null across all low-k bands, **STOP and report** "spray forecast-redundant"; the build below is moot. If GO, continue.

---

### Task 4: `peripheral_measurement` — the conditional-prior `(z', S')` (spec §4.2, load-bearing)

**Files:**
- Modify: `src/talent3.py` — add `peripheral_measurement`
- Test: `tests/test_talent3.py`

Given one season's measured channels `m` (D-vector: `m[0]`=xwOBA rate, `m[1:]`=peripheral means), block-diagonal measurement noise `S` (D×D), reference means `mu` (`mu[0]`=causal `μ_t`, `mu[1:]`=population peripheral means), and population **talent** covariance `Sigma` (D×D from `mvn_mle`), return `(z', S')` — an **unbiased** measurement of season xwOBA talent that borrows through `Σ_θp` and reduces to `(m[0], S[0,0])` when `Σ_θp=0`.

Derivation (spec §4.2): with `p|θ ~ N(μ_p + β(θ−μ_x), Σ_{pp·θ})`, `β=Σ_pθ/Σ_θθ`, and θ flat, the season likelihood is `N(θ; z', S')` with
`S' = 1/(1/S_xx + β'W⁻¹β)`, `z' = S'·[m_x/S_xx + (β'W⁻¹β)·μ_x + β'W⁻¹(m_p−μ_p)]`, `W = Σ_{pp·θ}+S_pp`. (Verified unbiased: `E[z'|θ]=θ`.)

- [ ] **Step 1: Write failing tests** — three, matching spec §4.4/§10:

```python
# tests/test_talent3.py
import numpy as np
from src.talent3 import peripheral_measurement

def _brute_force(m, S, mu, Sigma, grid):
    # marginal likelihood of theta via the conditional prior p(p|theta), on a grid
    Sxx, Spp, Sxp = Sigma[0,0], Sigma[1:,1:], Sigma[1:,0]
    beta = Sxp / Sxx; Sig_ppth = Spp - np.outer(Sxp, Sxp)/Sxx
    ll = []
    for th in grid:
        mp_mean = mu[1:] + beta*(th - mu[0]); W = Sig_ppth + S[1:,1:]
        # m_x|th ~ N(th, S00); m_p|th ~ N(mp_mean, W)   (block-diag S)
        lx = -0.5*((m[0]-th)**2/S[0,0])
        r = m[1:]-mp_mean; lp = -0.5*(r @ np.linalg.solve(W, r))
        ll.append(lx+lp)
    ll = np.array(ll); w = np.exp(ll-ll.max()); w/=w.sum()
    zt = (w*grid).sum(); vt = (w*(grid-zt)**2).sum()
    return zt, vt

def test_peripheral_measurement_matches_brute_force():
    Sigma = np.array([[4e-4, 6e-3, 1e-3],[6e-3, 4.0, 0.2],[1e-3, 0.2, 1e-2]])
    S = np.diag([2e-4, 1.5, 5e-3])                 # block-diagonal
    m = np.array([0.34, 92.0, 0.09]); mu = np.array([0.31, 89.0, 0.06])
    z1, S1 = peripheral_measurement(m, S, mu, Sigma)
    grid = np.linspace(0.2, 0.5, 60001)
    z2, S2 = _brute_force(m, S, mu, Sigma, grid)
    assert abs(z1-z2) < 1e-4 and abs(S1-S2) < 1e-6

def test_peripheral_measurement_reduces_to_rung_a():
    Sigma = np.diag([4e-4, 4.0, 1e-2])             # Sigma_theta_p = 0
    S = np.diag([2e-4, 1.5, 5e-3]); m = np.array([0.34, 92., .09]); mu = np.array([.31, 89., .06])
    z, Sp = peripheral_measurement(m, S, mu, Sigma)
    assert abs(z - m[0]) < 1e-12 and abs(Sp - S[0,0]) < 1e-12

def test_peripheral_measurement_unbiased():
    # draw theta, then peripherals from p(p|theta), then noisy m; z' should track theta
    rng = np.random.default_rng(0)
    Sigma = np.array([[4e-4, 8e-3],[8e-3, 4.0]]); S = np.diag([3e-4, 1.5])
    mux, mup = 0.31, 89.0; beta = Sigma[1:,0]/Sigma[0,0]
    Sig_ppth = Sigma[1:,1:]-np.outer(Sigma[1:,0],Sigma[1:,0])/Sigma[0,0]
    ths = rng.normal(mux, np.sqrt(Sigma[0,0]), 6000); zs = []
    for th in ths:
        p = rng.normal(mup + beta*(th-mux), np.sqrt(Sig_ppth[0,0]))
        m = np.array([th + rng.normal(0, np.sqrt(S[0,0])), p + rng.normal(0, np.sqrt(S[1,1]))])
        zs.append(peripheral_measurement(m, S, np.array([mux,mup]), Sigma)[0])
    zs = np.array(zs)
    # Attenuation toward mu_x has ~0 unconditional mean bias (theta is symmetric around mu_x),
    # so the mean check alone is blind to it. Regress z' on theta: an unbiased, non-shrinking
    # measurement has SLOPE ~ 1; a shrink toward mu_x depresses the slope below 1.
    slope = float(np.polyfit(ths, zs, 1)[0])
    assert abs(slope - 1.0) < 0.03            # catches attenuation directly
    assert abs(np.mean(zs - ths)) < 3e-4      # catches constant-offset / dropped-mu_x bugs
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_talent3.py -k peripheral_measurement -v` → FAIL.
- [ ] **Step 3: Implement**

```python
def peripheral_measurement(m, S, mu, Sigma):
    """Conditional-prior peripheral measurement of xwOBA talent (spec §4.2). Returns an
    UNBIASED (z', S') that borrows through the talent cross-cov Sigma[1:,0]; reduces to
    (m[0], S[0,0]) when that is zero. m[0]/mu[0]=xwOBA & causal mu_t; [1:]=peripherals."""
    m = np.asarray(m, float); mu = np.asarray(mu, float); Sigma = np.asarray(Sigma, float)
    Sxx = float(S[0, 0])
    if m.shape[0] == 1:                                   # xwOBA-only -> rung a
        return float(m[0]), Sxx
    beta = Sigma[1:, 0] / Sigma[0, 0]
    Sig_ppth = Sigma[1:, 1:] - np.outer(Sigma[1:, 0], Sigma[1:, 0]) / Sigma[0, 0]
    W = Sig_ppth + S[1:, 1:]
    Winv = np.linalg.inv(W)
    Ip = float(beta @ Winv @ beta)
    S_prime = 1.0 / (1.0 / Sxx + Ip)
    z_prime = S_prime * (m[0] / Sxx + Ip * mu[0] + beta @ Winv @ (m[1:] - mu[1:]))
    return float(z_prime), float(S_prime)
```

- [ ] **Step 4: Run to verify pass** — all three PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(talent3): conditional-prior peripheral_measurement (spec §4.2)"`

---

### Task 5: Fit the population talent covariance `Σ` and peripheral means (k-channel `mvn_mle`)

**Files:**
- Modify: `scripts/run_talent3.py` — extend `precompute_full_measurements` to k channels; add `fit_peripheral_hypers`
- Test: `tests/test_talent3.py` (or a script-level test module)

`peripheral_measurement` needs `Σ` (D×D talent cov) and population peripheral means `μ_p`. Fit them once via `talent2.mvn_mle` on **full-season** k-channel measurements of the stable population (`D_obs ≥ FIT_MIN_PA`), standardized (as `build_talent2_table` does), on the causally-allowed seasons. `mvn_mle` is already dimension-general — feed it the D-wide `z`/`S`.

- [ ] **Step 1: Write failing test** — 4-channel `mvn_mle` on data whose first 3 dims match a 3-channel fit reproduces the 3-channel `Σ` sub-block within tolerance (dimension-generality regression); `Σ` is PSD.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — make `precompute_full_measurements` **channel-gated**: for `channels=("xwoba",)` it routes through the current scalar `sample_measurement` path **unchanged** (rung a stays bit-identical — see Task 7); for peripheral channel sets it returns, per `(batter,season)`, the channel vector `m` (D,) and covariance `S` (D,D) via the generalized `bootstrap_S` (Task 2, `pull=` set) over full-season PAs (reuse `build_pa_frame`'s ev/barrel/pull). `fit_peripheral_hypers(full, channels, fit_seasons)` standardizes and calls **`mvn_mle` directly** — do **NOT** reuse `build_talent2_table` (it hard-asserts `dims[0]=="xwoba"` and does `DIMS.index("pull")`, which throws) — returning de-standardized per-season `μ` and `Σ`, plus `center/scale`. Keep the RNG-stream discipline (`sorted` iteration) intact.
  - **μ composition (locked):** current-season xwOBA slot uses causal `mu_kt`; prior-season xwOBA slots keep `mu_full[s]` (as rung a demeans today); **peripheral slots use the population `μ_p` from the `mvn_mle` fit** — not the fit's per-season `μ[s][0]`. Unbiasedness makes the xwOBA `μ_x` cancel, but wiring the wrong `μ` here is the easiest slip.
- [ ] **Step 4: Run → pass; regression** — 3-channel talent2 tests unaffected.
- [ ] **Step 5: Commit** — `git commit -am "feat(talent3): k-channel full-season measurements + Sigma fit"`

---

### Task 6: Wire rung-b measurement into the forecast (reduction identity end-to-end)

**Files:**
- Modify: `scripts/run_talent3.py` — `build_priors`, `forecast_row` (and `cutpoint_split` in `src/talent3.py` to expose observed ev/barrel/pull)
- Test: `tests/test_talent3.py`

Replace the scalar `sample_measurement` call **only when a peripheral channel set is active**. For each season (prior full + current first-*k*): assemble the k-channel `m` and `S`, call `peripheral_measurement` → `(z', S')`, and feed `z'/S'` into the exact same `z/mu/S` arrays consumed by `cutpoint_posterior`. `mu` for the current season keeps causal `mu_kt` in slot 0 and population `μ_p` in the peripheral slots. Rung a (`channels=("xwoba",)`) takes the current scalar path unchanged.

- [ ] **Step 1: Write failing test** — two reduction checks:
  - **(a) Strict bit-identity via the scalar route.** `channels=("xwoba",)` reproduces the rung-a `theta_hat/V/r_final_model` **bit-identically** on a toy player. This is the **only** bit-identical route.
  - **(b) Peripheral-route reduction (close, not exact).** With active peripheral channels but a fitted `Σ` whose `Σ_θp=0`, `forecast_row`'s `theta_hat` matches rung a only to `atol≈1e-9` — **not** bit-identically — because the k-channel `bootstrap_S` computes `S[0,0]` via `np.cov` over the all-channels-valid replicate subset, which differs at ULP (and materially for low-BBE players) from the scalar `.var`. Do not assert exact equality here.
  - Also: the leakage guard still fires on a future-dated conditioning PA.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — when `channels=("xwoba",)`, **skip `peripheral_measurement`** and use the scalar `(z, S[0,0])` exactly as today (preserves bit-identity). For peripheral sets: extend `cutpoint_split` to also slice `ev/barrel/pull` for first-*k* and rest; thread a `channels`/`hypers` argument through `run_sweep → forecast_row`; build `m`,`S` per season (block-diagonal `S`), call `peripheral_measurement`. Prior-season `(z',S')` are precomputed once (memoized, no RNG) in Task 5's structures.
- [ ] **Step 4: Run → pass; full regression** — `.venv/bin/python -m pytest tests/ -v` all green.
- [ ] **Step 5: Commit** — `git commit -am "feat(talent3): rung-b peripheral measurement wired into forecast_row"`

---

### Task 7: `--rung`/`--channels` flag and the three-model ablation

**Files:**
- Modify: `scripts/run_talent3.py` — `main`, `run_sweep` orchestration
- Test: `tests/test_talent3.py` (CLI/arg parsing) + a fast smoke

Add `--channels` (repeatable) with presets: `a` → `("xwoba",)`; `b_evbarrel` → `("xwoba","avg_ev","barrel_rate")`; `b_full` → `+("pull",)`. Default runs all three, writing `forecast_table_<tag>.parquet` per config; `--rung a` reproduces the shipped run.

- [ ] **Step 1: Failing test** — arg parsing yields the three channel tuples; `--rung a` maps to `("xwoba",)`.
- [ ] **Step 2 → 4:** implement, run, and a **regression smoke**: `--rung a` (which maps to `channels=("xwoba",)` → the scalar measurement path) forecast table equals the committed `results/talent3/forecast_table.parquet` on a fixed seed for a small season subset, **bit-identically**. If it moves, the cause is almost always rung a being routed through a non-scalar path (per Task 5/6) — fix the routing, don't loosen the check.
- [ ] **Step 5: Commit** — `git commit -am "feat(talent3): three-model ablation (--channels), rung-a reproduced"`

---

### Task 8: Scoring — three-model gate panel + the +pull/−pull marginal delta

**Files:**
- Modify: `scripts/run_talent3.py` — `run_scoring`, `cluster_boot_delta`
- Test: `tests/test_talent3.py`

Score all three tables on the existing panel (`rmse_by_band`, `coverage_by_band`, gates G1–G5 via the current machinery), then add the **paired** `cluster_boot_delta` between `b_full` and `b_evbarrel` (spray's marginal value) and between `b_full` and rung a (primary gate: RMSE at low PA-seen, CI excludes 0). **`cluster_boot_delta` compares two columns within one frame** (`src/talent3.py:321`), so first **join** the three per-model tables on `(batter, season, k)` into one frame (suffix each model's `r_final_model`) before calling it. Extend `metrics.json` with a `rung_b` block: per-model by-band RMSE + coverage, the two paired deltas with CIs, and the calibration-by-k for rung a vs b.

- [ ] **Step 1: Failing test** — on a synthetic three-table fixture, the primary-gate delta and the +pull/−pull delta are computed with correct sign and CI keys; schema present.
- [ ] **Step 2 → 4:** implement, run, pass. Preserve the parent's "report, don't hard-fail calibration" stance (spec §6).
- [ ] **Step 5: Commit** — `git commit -am "feat(talent3): rung-b three-model scoring + marginal-value bootstrap"`

---

### Task 9: Figures, NOTES, RESULTS

**Files:**
- Modify: `scripts/run_talent3.py` — a calibration figure rung a vs b at short runway
- Modify: `results/talent3/NOTES.md`, `results/RESULTS.md`

- [ ] Add a figure overlaying rung-a vs rung-b 50/80/90 coverage across k (short-runway focus). Reuse `fig_calibration_by_band` style.
- [ ] Draft the rung-b `NOTES.md` section and a `RESULTS.md` block: the three-model table, the two paired deltas, the calibration movement, honest limitations (redundancy, BBE-thin early). Numbers filled from Task 10's committed artifacts, not prose. Commit.

---

### Task 10: Full run, gate evaluation, write-up, commit

**Files:** `results/talent3/` artifacts; `NOTES.md`/`RESULTS.md` numbers

- [ ] **Pre-check** (Task 3) on real data → confirm GO. Record `precheck_pull.json`.
- [ ] Run the three-model sweep (closed-form, ~seconds–minutes; no BART): `.venv/bin/python scripts/run_talent3.py` (default = all three).
- [ ] Evaluate: **primary hard gate** — `b_full` beats rung a on RMSE at low PA-seen, paired-bootstrap CI excludes 0? Report the +pull/−pull marginal delta and whether the **short-runway calibration** miss narrows (reported, not gated). Confirm the §4.4 reduction identity test and leakage guard are green.
- [ ] Fill `NOTES.md`/`RESULTS.md` numbers against the committed `metrics.json` (guard-cell discipline). If the gate **fails** or spray's marginal delta is null, write it up as a documented negative (spec §6) — that is a valid outcome.
- [ ] Commit results (small artifacts + metrics; keep any large `.parquet` per the repo's gitignore convention). `git commit -m "results(talent3): rung-b spray peripheral — <verdict>"`

---

## Notes for the executor
- **DRY:** reuse `bootstrap_S`, `mvn_mle`, `cutpoint_posterior`, `forward_forecast`, `cluster_boot_delta`, `_spray_cols` — do not reimplement.
- **TDD:** every code task is red→green→commit; use @superpowers-extended-cc:test-driven-development.
- **The reduction tests (Tasks 4 & 6) are the safety net** — a wrong measurement fails them. Do not weaken their tolerances to pass.
- **Determinism:** preserve the `sorted()` iteration + shared-RNG discipline (`results/talent3/NOTES.md` gotcha; and the memory note on polars needing explicit `.sort()`).
- **Two EB fits, two roles:** `Σ` (Task 5) parameterizes only the peripheral→θ *borrow*; `(σ_η², σ_u²)` (existing `fit_hypers_eb`) remain θ's hierarchical prior. When `Σ_θp=0` the measurement ignores `Σ` and the hierarchy fully controls shrinkage.
