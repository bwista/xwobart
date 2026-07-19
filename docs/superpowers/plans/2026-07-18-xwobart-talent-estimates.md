# xwobart Talent Estimates Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the raw per-player xwOBA band with a **true-talent estimate** per batter-season — a shrunk center plus a calibrated interval that narrows with PA and is centered on where the hitter's true talent actually is — via empirical Bayes now (Phase 1), and lay out a BART-informed hierarchical model as an explicit later step (Phase 2).

**Architecture:** Phase 1 is a Gaussian–Gaussian empirical-Bayes (James–Stein / Efron–Morris) shrinkage layer over quantities we already have: each player-season's raw xwOBA and its sampling SE, plus the season population. It produces `θ̂ = μ + reliability·(raw − μ)` with `reliability = τ²/(τ² + SE²)` and a posterior interval `θ̂ ± z·√(post_var)`. No BART, no re-fit. Phase 2 replaces the flat league-mean prior with a BART contact-quality prior (a player with elite contact but few PAs shrinks toward their contact-implied xwOBA, not toward the league) — a separate cycle that requires persisting per-event model EVs and one re-fit.

**Tech Stack:** Python 3.12, Polars (pandas only at boundaries), NumPy, matplotlib, pytest. Reuses `src.config`, the slim Statcast caches in `data/raw/`, and `results/stage_C/player_table.parquet`. All Phase-1 math is pure and unit-tested; the orchestrator is verified against real data.

**Background (why this plan exists):** Established this session and recorded in `results/RESULTS.md`:
- v0's BART posterior interval does **not** shrink with PA (`results/task_a/`) — it measures surface uncertainty at a contact profile, not sample size. Wrong object for "how good is this hitter."
- v0 is at **statistical parity** with Savant at predicting next-season wOBA (`results/benchmark/`, r 0.481 vs 0.487) — it does not beat Savant, and the 3-feature model is at Savant's information ceiling.
- A **bootstrap over a player's PAs** (`results/player_ci/`) gives a band that correctly narrows with PA — but it is centered on the *raw* number, so it over-credits hot small samples (e.g. Trout 2024, 125 PA: raw .407, band [.336,.480], but reliability-regressed true talent ≈ .363).
- This plan builds the estimate that is actually useful for analyzing batters: a **sample-size-regressed true-talent xwOBA with a calibrated interval**. Its payoff is directly testable — a shrunk estimate should predict next-season wOBA **better than raw Savant xwOBA**, especially for low-PA players (a real path to beating the r 0.487 anchor without new features).

**Worktree note:** Per the repo convention (see `docs/superpowers/handoffs/2026-07-18-v1-handoff.md`), work happens directly on `main` — no worktree. Commit per task; push only when asked.

**Skills:** @superpowers-extended-cc:test-driven-development for every TDD task; @superpowers-extended-cc:verification-before-completion before claiming any task done; @superpowers-extended-cc:systematic-debugging when anything fails.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/talent.py` | Pure logic: per-PA xwOBA values, per-player raw xwOBA + sampling SE, empirical-Bayes hyperparameter fit, shrinkage estimate + interval, season-level assembly |
| `tests/test_talent.py` | Unit tests for every pure function in `src/talent.py` (synthetic data, known answers) |
| `scripts/run_talent.py` | Orchestrator: build the talent table from the slim caches, join names + Savant, write `results/talent/`, render figures, run the EB-vs-raw-vs-Savant predictive validation |
| `results/talent/` (generated) | `talent_table.parquet`, `talent_metrics.json`, `figures/*.png`, `NOTES.md` |
| `results/RESULTS.md` | Add a "Talent estimates (empirical Bayes)" section |

Data flow: slim caches → `build_pa_values` → `per_player_raw` → `eb_fit` (per season) → `eb_shrink` → `build_talent_table` → `run_talent.py` (figures + validation + docs).

**Reuse note (DRY):** `scripts/player_ci_bootstrap.py` already has a working per-PA value construction; Phase 1 promotes that logic into `src.talent.build_pa_values` and both the new orchestrator and (optionally) the existing bootstrap script use the shared function. The predictive-validation task reuses `scripts/benchmark_vs_savant.py`'s `actual_woba`/`_pearson`/`_calibrated_rmse` helpers.

---

## Phase 1 — Empirical-Bayes true-talent xwOBA (build now)

### Task 1: Per-PA values and per-player raw xwOBA + sampling SE (TDD)

**Files:**
- Create: `src/talent.py`
- Test: `tests/test_talent.py`

- [ ] **Step 1.1: Write the failing tests**

`tests/test_talent.py`:
```python
import numpy as np
import polars as pl

from src.talent import build_pa_values, per_player_raw


def _pitches():
    # 2 players, 2024. type X uses est_woba; walk/K use woba_value; non-PA rows dropped.
    return pl.DataFrame({
        "batter":       [1, 1, 1, 2, 2, 1],
        "game_year":    [2024, 2024, 2024, 2024, 2024, 2024],
        "type":         ["X", "X", "B", "X", "S", "S"],
        "estimated_woba_using_speedangle": [1.2, 0.1, None, 0.8, None, None],
        "woba_value":   [1.25, 0.0, 0.69, 0.9, 0.0, 0.0],
        "woba_denom":   [1, 1, 1, 1, 1, None],   # last row is a non-PA pitch -> dropped
    })


def test_build_pa_values_picks_est_for_bbe_and_woba_else():
    pav = build_pa_values(_pitches()).sort("batter", "value")
    # player 1: three PAs -> values 1.2 (X est), 0.1 (X est), 0.69 (walk woba_value)
    v1 = pav.filter(pl.col("batter") == 1)["value"].sort().to_list()
    assert v1 == [0.1, 0.69, 1.2]
    # non-PA row (woba_denom null) dropped: total 4 PAs across both players
    assert pav.height == 4
    assert set(pav.columns) == {"batter", "season", "value", "denom"}


def test_per_player_raw_xwoba_and_se():
    pav = build_pa_values(_pitches())
    raw = per_player_raw(pav).sort("batter")
    r1 = raw.filter(pl.col("batter") == 1).row(0, named=True)
    assert r1["PA"] == 3
    assert abs(r1["xwoba_raw"] - (1.2 + 0.1 + 0.69) / 3) < 1e-9
    # se = sample sd of the three values / sqrt(3)
    vals = np.array([1.2, 0.1, 0.69])
    assert abs(r1["se"] - vals.std(ddof=1) / np.sqrt(3)) < 1e-9
    assert r1["se2"] > 0
```

- [ ] **Step 1.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_talent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.talent'`

- [ ] **Step 1.3: Implement `src/talent.py`**

```python
"""True-talent xwOBA via empirical-Bayes shrinkage (spec: 2026-07-18 talent plan).

Each player-season's raw xwOBA is a noisy estimate of their true talent; shrink it
toward the season population by its reliability (a function of sample size), giving a
center that regresses small samples toward the mean and an interval that narrows with
PA. Pure functions only — orchestration lives in scripts/run_talent.py."""
from __future__ import annotations

import numpy as np
import polars as pl

Z90 = 1.6448536269514722  # 90% two-sided normal quantile


def build_pa_values(pitches: pl.DataFrame) -> pl.DataFrame:
    """One row per plate appearance: (batter, season, value, denom). A PA's xwOBA
    value is the batted-ball estimate for BBE (est_woba, falling back to the
    deterministic woba_value if missing) and woba_value for walks/Ks/HBP. Only
    PA-ending rows (woba_denom not null) count."""
    return (
        pitches.filter(pl.col("woba_denom").is_not_null())
        .with_columns(
            value=pl.when(pl.col("type") == "X")
            .then(pl.coalesce("estimated_woba_using_speedangle", "woba_value"))
            .otherwise(pl.col("woba_value"))
        )
        .select("batter", season="game_year", value="value", denom="woba_denom")
    )


def per_player_raw(pav: pl.DataFrame) -> pl.DataFrame:
    """Per (batter, season): raw xwOBA = Σvalue/Σdenom, PA, and the sampling SE of
    that mean (sd of per-PA values / √n). se2 is the variance used by the EB fit."""
    return (
        pav.group_by("batter", "season")
        .agg(
            PA=pl.col("denom").sum().cast(pl.Int64),
            n=pl.len(),
            num=pl.col("value").sum(),
            den=pl.col("denom").sum(),
            sd=pl.col("value").std(ddof=1),
        )
        .with_columns(
            xwoba_raw=pl.col("num") / pl.col("den"),
            se=pl.col("sd") / pl.col("n").sqrt(),
        )
        .with_columns(se2=pl.col("se") ** 2)
        .drop("num", "den", "sd")
        .sort("batter", "season")
    )
```

- [ ] **Step 1.4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_talent.py -v`
Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/talent.py tests/test_talent.py
git commit -m "feat: per-PA xwOBA values and per-player raw xwOBA + sampling SE"
```

---

### Task 2: Empirical-Bayes fit and shrinkage (TDD)

**Files:**
- Modify: `src/talent.py`
- Test: `tests/test_talent.py` (append)

- [ ] **Step 2.1: Write the failing tests (append)**

```python
from src.talent import eb_fit, eb_shrink


def test_eb_fit_recovers_hyperparameters():
    # Simulate true talents ~ N(0.32, 0.05^2); noisy observations with known SEs.
    rng = np.random.default_rng(0)
    n = 4000
    theta = rng.normal(0.32, 0.05, n)
    se = rng.uniform(0.02, 0.08, n)            # heteroscedastic (small vs large samples)
    raw = theta + rng.normal(0, se)
    mu, tau2 = eb_fit(raw, se ** 2)
    assert abs(mu - 0.32) < 0.005
    assert abs(np.sqrt(tau2) - 0.05) < 0.01


def test_eb_shrink_monotone_and_centered():
    mu, tau2 = 0.320, 0.05 ** 2
    raw = np.array([0.400, 0.400])             # same raw, different precision
    se2 = np.array([0.02 ** 2, 0.08 ** 2])     # small SE (many PA) vs large SE (few PA)
    theta, pv, lo, hi, rel = eb_shrink(raw, se2, mu, tau2)
    # smaller SE -> higher reliability -> less shrinkage -> theta closer to raw
    assert rel[0] > rel[1]
    assert theta[0] > theta[1]
    assert mu < theta[1] < raw[1]              # shrinks toward mu, never past it
    # posterior interval is narrower for the more reliable (smaller-SE) estimate
    assert (hi[0] - lo[0]) < (hi[1] - lo[1])
    # posterior variance = tau2*se2/(tau2+se2)
    assert np.allclose(pv, tau2 * se2 / (tau2 + se2))


def test_eb_shrink_edge_zero_tau():
    # No between-player spread -> full shrink to mu, degenerate interval floored, not NaN.
    theta, pv, lo, hi, rel = eb_shrink(np.array([0.5]), np.array([0.01]), 0.32, 0.0)
    assert theta[0] == 0.32 and rel[0] == 0.0 and np.isfinite(lo[0]) and np.isfinite(hi[0])
```

- [ ] **Step 2.2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_talent.py -v` — Expected: 3 new FAIL (ImportError).

- [ ] **Step 2.3: Implement (append to `src/talent.py`)**

```python
def eb_fit(raw: np.ndarray, se2: np.ndarray, tau2_floor: float = 1e-8) -> tuple[float, float]:
    """Gaussian–Gaussian empirical Bayes by method of moments (DerSimonian–Laird
    style). Observed variance of raw = between-player τ² + mean within-player SE²;
    μ is the precision-weighted mean. Returns (mu, tau2)."""
    raw = np.asarray(raw, float)
    se2 = np.asarray(se2, float)
    grand = raw.mean()
    tau2 = max(float(((raw - grand) ** 2).mean() - se2.mean()), tau2_floor)
    w = 1.0 / (tau2 + se2)
    mu = float((w * raw).sum() / w.sum())
    # one refinement of tau2 around the weighted mean
    tau2 = max(float((w * ((raw - mu) ** 2 - se2)).sum() / w.sum()), tau2_floor)
    return mu, tau2


def eb_shrink(raw: np.ndarray, se2: np.ndarray, mu: float, tau2: float):
    """Posterior for each player's true talent under N(mu, tau2) prior and N(theta, se2)
    likelihood. Returns (theta_hat, post_var, ci_lo, ci_hi, reliability)."""
    raw = np.asarray(raw, float)
    se2 = np.asarray(se2, float)
    rel = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    theta = mu + rel * (raw - mu)
    post_var = rel * se2                      # = tau2*se2/(tau2+se2); 0 when tau2==0
    half = Z90 * np.sqrt(post_var)
    return theta, post_var, theta - half, theta + half, rel
```

- [ ] **Step 2.4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_talent.py -v` — Expected: 5 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/talent.py tests/test_talent.py
git commit -m "feat: empirical-Bayes hyperparameter fit and shrinkage estimate"
```

---

### Task 3: Season-level talent table assembly (TDD)

**Files:**
- Modify: `src/talent.py`
- Test: `tests/test_talent.py` (append)

Fits EB hyperparameters **per season** (league offense varies year to year) on a stable population (`fit_min_pa`, default 100), then applies the shrinkage to **every** player-season including small samples (the small samples are exactly where the estimate is most useful).

- [ ] **Step 3.1: Write the failing test (append)**

```python
from src.talent import build_talent_table


def test_build_talent_table_per_season_and_small_samples_shrink_more():
    rng = np.random.default_rng(1)
    rows = []
    for season, lg in ((2023, 0.33), (2024, 0.31)):
        for batter in range(300):
            pa = int(rng.integers(50, 650))
            theta = rng.normal(lg, 0.045)
            vals = rng.normal(theta, 0.9, pa)            # per-PA values, high variance
            for v in vals:
                rows.append({"batter": batter, "game_year": season,
                             "type": "X", "estimated_woba_using_speedangle": v,
                             "woba_value": v, "woba_denom": 1})
    pav = build_pa_values(pl.DataFrame(rows))
    tbl = build_talent_table(pav, fit_min_pa=100)
    assert {"batter", "season", "PA", "xwoba_raw", "xwoba_talent",
            "talent_lo", "talent_hi", "reliability", "mu_season"}.issubset(tbl.columns)
    # per-season mu differs and is near the simulated league levels
    mus = dict(tbl.group_by("season").agg(pl.col("mu_season").first()).iter_rows())
    assert abs(mus[2023] - 0.33) < 0.02 and abs(mus[2024] - 0.31) < 0.02
    # low-PA players are shrunk harder (reliability rises with PA)
    lo = tbl.filter(pl.col("PA") < 120)["reliability"].mean()
    hi = tbl.filter(pl.col("PA") > 500)["reliability"].mean()
    assert hi > lo
    # talent estimate lies between raw and its season mean (shrinkage direction)
    s = tbl.filter(pl.col("PA") < 120)
    pulled = ((s["xwoba_talent"] - s["mu_season"]).abs()
              <= (s["xwoba_raw"] - s["mu_season"]).abs() + 1e-9).all()
    assert pulled
```

- [ ] **Step 3.2: Run to verify failure** — Expected: FAIL (ImportError).

- [ ] **Step 3.3: Implement (append to `src/talent.py`)**

```python
def build_talent_table(pav: pl.DataFrame, fit_min_pa: int = 100) -> pl.DataFrame:
    """Per (batter, season): raw xwOBA, EB true-talent estimate, 90% interval, and
    reliability. EB hyperparameters (mu, tau2) are fit per season on players with
    PA >= fit_min_pa, then applied to all player-seasons. Single-PA seasons have no
    sampling SD (se2 null) and are dropped — they carry no estimable uncertainty."""
    raw = per_player_raw(pav).filter(pl.col("se2").is_not_null())
    out = []
    for season in raw["season"].unique().sort().to_list():
        s = raw.filter(pl.col("season") == season)
        fit = s.filter(pl.col("PA") >= fit_min_pa)
        mu, tau2 = eb_fit(fit["xwoba_raw"].to_numpy(), fit["se2"].to_numpy())
        theta, pv, lo, hi, rel = eb_shrink(
            s["xwoba_raw"].to_numpy(), s["se2"].to_numpy(), mu, tau2
        )
        out.append(s.with_columns(
            xwoba_talent=pl.Series(theta),
            talent_post_var=pl.Series(pv),
            talent_lo=pl.Series(lo),
            talent_hi=pl.Series(hi),
            reliability=pl.Series(rel),
            mu_season=pl.lit(mu),
            tau_season=pl.lit(float(np.sqrt(tau2))),
        ))
    return pl.concat(out).sort("batter", "season")
```

- [ ] **Step 3.4: Run to verify pass** — Expected: 6 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/talent.py tests/test_talent.py
git commit -m "feat: per-season empirical-Bayes talent table assembly"
```

---

### Task 4: Orchestrator — build the real talent table + figures

**Files:**
- Create: `scripts/run_talent.py`

No unit test (thin orchestration over real files); verified by running against the caches.

- [ ] **Step 4.1: Implement `scripts/run_talent.py`**

```python
"""Build the empirical-Bayes true-talent xwOBA table from the slim Statcast caches,
join display names + public Savant xwOBA, render figures, and write results/talent/.
No re-fit. Run from repo root: `.venv/bin/python scripts/run_talent.py`."""
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

from src.config import load_config
from src.talent import build_pa_values, build_talent_table

C_RAW = "#8a8a8a"; C_TAL = "#4878CF"; C_REF = "#8a8a8a"


def load_pitches(cfg, seasons) -> pl.DataFrame:
    return pl.concat([
        pl.read_parquet(cfg.raw_dir / f"statcast-{y}-slim.parquet") for y in seasons
    ])


def fig_shrinkage(figdir, tbl):
    """Raw -> talent for 2024, showing small samples pulled toward the mean."""
    d = tbl.filter(pl.col("season") == 2024)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(d["PA"], d["xwoba_raw"], s=10, alpha=0.3, color=C_RAW, label="raw xwOBA")
    ax.scatter(d["PA"], d["xwoba_talent"], s=10, alpha=0.5, color=C_TAL, label="EB true-talent")
    ax.axhline(d["mu_season"][0], color=C_REF, ls="--", lw=1, label="season mean")
    ax.set_xlabel("PA"); ax.set_ylabel("xwOBA"); ax.set_xscale("log")
    ax.set_title("Shrinkage: small samples pulled toward the mean (2024)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "shrinkage_raw_to_talent.png", dpi=120); plt.close(fig)


def fig_reliability(figdir, tbl):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tbl["PA"], tbl["reliability"], s=8, alpha=0.2, color=C_TAL)
    ax.set_xlabel("PA"); ax.set_ylabel("reliability  τ²/(τ²+SE²)")
    ax.set_title("Reliability rises with sample size")
    ax.grid(True, color=C_REF, alpha=0.15)
    fig.tight_layout(); fig.savefig(figdir / "reliability_vs_pa.png", dpi=120); plt.close(fig)


def main():
    cfg = load_config()
    seasons = cfg.all_seasons
    pav = build_pa_values(load_pitches(cfg, seasons))
    tbl = build_talent_table(pav, fit_min_pa=cfg.min_pa)

    # names + public Savant xwOBA from the frozen v0 player table
    pt = pl.read_parquet(cfg.results_dir / "stage_C" / "player_table.parquet").select(
        "batter", "season", "player_name", "xwoba_savant"
    )
    tbl = tbl.join(pt, on=["batter", "season"], how="left")

    outdir = cfg.results_dir / "talent"; figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_shrinkage(figdir, tbl); fig_reliability(figdir, tbl)
    tbl.write_parquet(outdir / "talent_table.parquet")

    metrics = {
        "n_player_seasons": tbl.height,
        "per_season": tbl.group_by("season").agg(
            n=pl.len(), mu=pl.col("mu_season").first(), tau=pl.col("tau_season").first(),
            median_reliability=pl.col("reliability").median(),
        ).sort("season").to_dicts(),
        "biggest_shrinks": tbl.with_columns(shrink=(pl.col("xwoba_raw") - pl.col("xwoba_talent")).abs())
            .sort("shrink", descending=True).head(15)
            .select("player_name", "season", "PA", "xwoba_raw", "xwoba_talent", "reliability").to_dicts(),
    }
    (outdir / "talent_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))
    print(f"  wrote {outdir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run against the real caches**

Run: `.venv/bin/python scripts/run_talent.py`
Expected: `results/talent/talent_table.parquet` (~2,668 player-seasons), two figures, `talent_metrics.json`. Per-season `mu` ≈ 0.31–0.33, `tau` ≈ 0.03–0.05, median reliability rising across seasons. `biggest_shrinks` should be hot/cold small-sample players (e.g. low-PA hitters with extreme raw xwOBA pulled sharply toward the mean).

- [ ] **Step 4.3: Eyeball the figures**

Open `results/talent/figures/shrinkage_raw_to_talent.png` and `reliability_vs_pa.png`. Confirm: raw points fan out at low PA and the talent points collapse toward the season mean there; reliability → 1 as PA grows. If a figure looks wrong, use @superpowers-extended-cc:systematic-debugging.

- [ ] **Step 4.4: Full suite still green** — `.venv/bin/pytest -q` — Expected: all pass.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/run_talent.py
git commit -m "feat: talent-table orchestrator with shrinkage + reliability figures"
```

---

### Task 5: Validation — does shrinkage beat raw Savant at predicting next season? (the payoff)

**Files:**
- Modify: `scripts/run_talent.py` (add a `validate()` step)
- Reuse: `scripts/benchmark_vs_savant.py` (`actual_woba`, `_pearson`, `_calibrated_rmse`)

This is the decisive check: EB true-talent xwOBA_T should predict actual wOBA_{T+1} **at least as well as, and ideally better than, raw Savant xwOBA_T** — most of the gain concentrated in low-PA players. If it clears Savant's r 0.487 anchor, we have beaten Savant by shrinkage alone.

- [ ] **Step 5.1: Implement `validate()` (append to `scripts/run_talent.py`, call from `main`)**

```python
def validate(cfg, tbl) -> dict:
    """Predict next-season actual wOBA from {EB talent, raw xwOBA, Savant} at season T."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from benchmark_vs_savant import actual_woba, _pearson, _calibrated_rmse

    act = actual_woba(cfg.raw_dir, cfg.all_seasons)
    base = tbl.select("batter", "season", "xwoba_talent", "xwoba_raw", "xwoba_savant", "PA").join(
        act.select("batter", "season", "actual_woba"), on=["batter", "season"], how="inner"
    )
    seasons = cfg.all_seasons
    rows = []
    for t in seasons[:-1]:
        a = base.filter(pl.col("season") == t)
        b = base.filter(pl.col("season") == t + 1).select("batter", target="actual_woba", pa_next="PA")
        j = a.join(b, on="batter", how="inner").filter(
            (pl.col("PA") >= cfg.min_pa) & (pl.col("pa_next") >= cfg.min_pa))
        rows.append(j)
    pairs = pl.concat(rows)
    tgt = pairs["target"].to_numpy()
    out = {"n_pairs": pairs.height}
    for pred in ("xwoba_talent", "xwoba_raw", "xwoba_savant"):
        p = pairs[pred].to_numpy()
        out[pred] = {"r": _pearson(p, tgt), "rmse_calibrated": _calibrated_rmse(p, tgt)}
    # low-PA subset (where shrinkage should help most)
    lowpa = pairs.filter(pl.col("PA") < 250)
    if lowpa.height >= 30:
        lt = lowpa["target"].to_numpy()
        out["lowPA_under250"] = {"n": lowpa.height, **{
            pred: {"r": _pearson(lowpa[pred].to_numpy(), lt)}
            for pred in ("xwoba_talent", "xwoba_raw", "xwoba_savant")}}
    out["beats_savant_pooled"] = out["xwoba_talent"]["r"] > out["xwoba_savant"]["r"]
    return out
```

Wire into `main`: `metrics["validation"] = validate(cfg, tbl)` and print it.

- [ ] **Step 5.2: Run and record**

Run: `.venv/bin/python scripts/run_talent.py`
Expected: `validation.xwoba_talent.r ≥ xwoba_raw.r`, and `xwoba_talent.r` within noise of / above `xwoba_savant.r` (0.487 anchor); the low-PA subset should show the clearest talent-vs-raw gap. Record the exact numbers — they go in RESULTS.md and the note.

**Interpretation gate:** if EB does **not** beat raw at even the low-PA subset, stop and use @superpowers-extended-cc:systematic-debugging — shrinkage improving small-sample prediction is the whole thesis; a null there means a bug in `se`/`reliability` or the fit population.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/run_talent.py
git commit -m "feat: next-season predictive validation of EB talent vs raw vs Savant"
```

---

### Task 6: Docs — talent note + RESULTS.md section

**Files:**
- Create: `results/talent/NOTES.md`
- Modify: `results/RESULTS.md`

- [ ] **Step 6.1: Write `results/talent/NOTES.md`**

Cover: the goal (true-talent estimate for analyzing batters), the construction (per-PA values → raw + SE → per-season EB shrinkage → interval), the shrinkage behavior (reliability vs PA; a couple of concrete hot/cold small-sample examples raw→talent), and the **validation result** (EB vs raw vs Savant on next-season wOBA, pooled + low-PA). State plainly whether EB beats/ties Savant and by how much. Note the two limitations carried into Phase 2: (1) the prior is the flat league mean — a good-contact/low-PA hitter is shrunk toward league, not toward their contact-implied xwOBA; (2) the interval is the estimation (talent) interval and does not yet fold in BART's surface term.

- [ ] **Step 6.2: Add a RESULTS.md section**

Add a "Talent estimates (empirical Bayes)" block after the existing player-CI section: the per-season μ/τ, median reliability, the validation table (talent/raw/Savant r + calibrated RMSE, pooled and low-PA), and the verdict vs the 0.487 anchor.

- [ ] **Step 6.3: Commit**

```bash
git add results/talent/NOTES.md results/RESULTS.md
git commit -m "docs: empirical-Bayes talent estimates — results and validation"
```

---

## Phase 2 — BART-informed hierarchical model (LATER; separate spec→plan→execute cycle)

> **Do not start Phase 2 inside this plan.** It requires a modeling decision and one BART re-fit, so it should be run as its own brainstorm→spec→plan→execute cycle (like v0). This section is the roadmap and the decisions to make, not bite-sized tasks.

**Why.** Phase 1 shrinks every player toward the *league mean*. That is wrong for a hitter whose *contact quality* says more than their small sample: a rookie with 80 PA of barrels should regress toward a barrel-hitter's xwOBA, not toward league average. Phase 2 replaces the flat prior mean with a **BART contact-quality prediction**, so the player's own PAs update a contact-informed prior. This is the job that finally gives the BART model a clear, defensible role (Phase 1 is model-agnostic; v0's posterior interval was the wrong object).

**Prerequisite (blocks everything in Phase 2):** persist **per-event model expected values** during a fit. The fitted BART trees are not saved in `idata.nc` (only in `model["mu"].owner.op.all_trees` in memory), so the model's per-BBE EVs cannot be recovered without a re-fit. Add an option to `scripts/run_v0.py` / `src/model.py` to write per-event `ev_mean` (holdout **and** train) to `results/stage_C/event_ev.parquet` keyed by (batter, season, event index). One re-fit of Stage C (~27 min) covers it.

**Design decision to settle in the Phase-2 brainstorm — two variants:**

1. **Two-stage (recommended first; cheaper, reuses v0).**
   - Stage 1: prior mean `m_i` = the player-season's **contact-implied xwOBA** = mean of the model's per-event EVs over their BBE + the deterministic non-BBE values (from the persisted `event_ev.parquet`).
   - Stage 2: hierarchical shrink of the raw xwOBA toward `m_i` (not the league mean): `θ̂_i = m_i + reliability_i·(raw_i − m_i)`, with between-player residual variance `τ_resid²` estimated by the same EB machinery on `(raw_i − m_i)`. Reuses all of `src.talent`.
   - Validation: should beat Phase-1 league-mean EB **specifically for players whose contact quality diverges from their small-sample results** (e.g. a hitter barreling the ball with unlucky outcomes over 100 PA).

2. **Full event-level hierarchical BART (heaviest; most principled).**
   - Add a **batter random intercept** to the latent categorical model in `src/model.py`: `mu = BART(contact features) + b_batter`, `b_batter ~ N(0, σ_b²)`. The model gains player identity and does partial pooling internally; per-player posteriors come straight out of the fit.
   - Cost: a genuinely new, larger fit (thousands of batter effects); needs its own runtime/memory gate like v0's Stage C decision. Only pursue if the two-stage variant proves the concept and the extra coherence is judged worth the fit cost.

**Combined interval (either variant).** Report the fullest interval as `width ≈ √(talent_var + surface_var)` — the Phase-1/Phase-2 estimation (talent) variance plus BART's surface posterior variance (the flat ≈0.056 term from v0). Sampling/estimation dominates at low PA; the BART surface term dominates at high PA. This is where v0's posterior finally contributes legitimately instead of masquerading as the whole band.

**Phase-2 definition of done (mirrors v0 §15):** the two-stage contact-prior estimate beats Phase-1 league-mean EB on next-season wOBA for the low-PA / contact-diverges subset (or a clear reason why not); `event_ev.parquet` persisted and documented; the combined-interval construction implemented and its calibration checked (does the interval contain next-season wOBA at the nominal 90%?); `results/RESULTS.md` updated with the model-comparison table; all unit tests green.

---

## Definition of done (Phase 1)

- `src/talent.py` fully unit-tested; `.venv/bin/pytest -q` green (existing 28 + the new talent tests).
- `results/talent/` produced: `talent_table.parquet` (raw + true-talent + interval + reliability per batter-season), two figures, `talent_metrics.json`, `NOTES.md`.
- The next-season validation is recorded with exact numbers, and RESULTS.md states plainly whether EB true-talent beats/ties raw Savant xwOBA (vs the r 0.487 anchor), pooled and for low-PA players.
- Phase 2 is documented as the explicit next cycle with its prerequisite (persist per-event EVs) called out.
