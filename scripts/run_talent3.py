"""Rest-of-season xwOBA forecast: the validation sweep (Task 9). Wires the pure
functions in src/talent3.py, src/forecast.py and src/benchmarks.py over real
Statcast PAs: for every eligible hitter-season and a set of mid-season cutpoints,
forecast the hitter's FINAL full-season xwOBA and race the two-level talent model
against naive / league-shrunk / marcel / single-season baselines.

Discipline that makes the sweep trustworthy: leave-one-season-out (LOSO)
hyperparameters so a target season's own rows never train the prior fit for it;
a causal leakage guard (assert_causal) on EVERY forecast so no conditioning PA
can post-date its cutpoint; full-season measurements bootstrapped ONCE and reused
for the LOSO fit and the priors; a single shared RNG for reproducibility, with all
iteration explicitly sorted (polars group_by / partition_by do not guarantee order).

Writes results/talent3/forecast_table.parquet + leakage_digest.json. Scoring/gates
are Task 10; NOTES/figures are Task 11. Run from repo root:
`.venv/bin/python scripts/run_talent3.py`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import polars as pl

from src.benchmarks import league_shrunk, marcel
from src.config import load_config
from src.forecast import final_line_blend, forward_forecast
from src.talent import eb_fit
from src.talent3 import (
    assert_causal,
    build_pa_frame,
    cutpoint_posterior,
    cutpoint_split,
    fit_hypers_eb,
    sample_measurement,
    season_mu_causal,
)
from run_talent import load_pitches

CUTPOINTS = [50, 100, 150, 200, 300]   # first-k PAs "observed" at the mid-season cut
MIN_REMAINING = 30                     # need a real rest-of-season (D_rest >= this)
B_BOOT = 500                           # bootstrap reps for a measurement's variance
B_FWD = 600                            # forward-bootstrap reps for the predictive range
FIT_MIN_PA = 100                       # min full-season denom to enter a hyper/tau fit


def precompute_full_measurements(groups: dict, rng: np.random.Generator) -> dict:
    """(z, s00, D) for every (batter, season) FULL-season sample, ONE bootstrap
    each on the shared rng. Reused by the LOSO fit, the tau2 fit and the priors, so
    it is never recomputed per cutpoint. Groups iterated in sorted order to pin the
    RNG stream."""
    full = {}
    for key in sorted(groups):
        g = groups[key]
        v = g["value"].to_numpy(); d = g["denom"].to_numpy()
        z, s00 = sample_measurement(v, d, B_BOOT, rng)
        full[key] = (z, s00, float(d.sum()))
    return full


def fit_tau2(full: dict, seasons: list[int]) -> dict:
    """Per-season between-player variance tau2 (talent.eb_fit) over full-season
    measurements with denom >= FIT_MIN_PA. Feeds the league_shrunk benchmark."""
    tau2 = {}
    for t in seasons:
        zs, ss = [], []
        for b in sorted(b for (b, s) in full if s == t):
            z, s00, D = full[(b, t)]
            if D >= FIT_MIN_PA:
                zs.append(z); ss.append(s00)
        _, tau2[t] = eb_fit(np.array(zs), np.array(ss))
    return tau2


def fit_loso_phi(full: dict, mu_full: dict, seasons: list[int]) -> dict:
    """Leave-one-season-out (sigma_eta^2, sigma_u^2). For target season t, fit the
    rung-a hierarchy on every OTHER season's full-season measurements (denom >=
    FIT_MIN_PA), demeaned by their full-season league mean. The target season's rows
    never train its own prior."""
    phi = {}
    for t in seasons:
        ys, ss, pid = [], [], []
        for (b, s) in sorted(full):
            if s == t:
                continue
            z, s00, D = full[(b, s)]
            if D < FIT_MIN_PA:
                continue
            ys.append(z - mu_full[s]); ss.append(s00); pid.append(b)
        se2, su2 = fit_hypers_eb(np.array(ys), np.array(ss), np.array(pid))
        phi[t] = (se2, su2)
        print(f"  phi[{t}] (LOSO, n={len(ys)}): "
              f"sigma_eta={np.sqrt(se2):.5f}, sigma_u={np.sqrt(su2):.5f}")
    return phi


def build_priors(i: int, t: int, all_seasons: list[int], groups: dict, full: dict,
                 mu_full: dict) -> tuple:
    """Prior completed seasons s<t for player i (uses no RNG -> safe to memoize):
    model triples (z_full, mu_full[s], S_full) ascending; marcel rates+denoms
    most-recent-first; and the concatenated prior PA frame for the forward-bootstrap
    reference multiset and the leakage guard (None when the player has no history)."""
    seasons = [s for s in all_seasons if s < t and (i, s) in groups]   # ascending
    model = [(full[(i, s)][0], mu_full[s], full[(i, s)][1]) for s in seasons]
    m_rates = [full[(i, s)][0] for s in reversed(seasons)]             # recent first
    m_denoms = [full[(i, s)][2] for s in reversed(seasons)]
    prior_pas = pl.concat([groups[(i, s)] for s in seasons]) if seasons else None
    return model, m_rates, m_denoms, prior_pas


def forecast_row(i: int, t: int, k: int, cp: dict, pas_sorted: pl.DataFrame,
                 priors: tuple, mu_kt: float, tau2_t: float, phi_t: tuple,
                 rng: np.random.Generator) -> tuple[dict, int]:
    """One forecast: assemble the two-level posterior on theta_{i,t}, forward
    -bootstrap the final-line predictive range, and compute the actual final line
    plus the five point benchmarks. Returns (row, conditioning-row count).

    mu scales (spec-locked): completed prior seasons demean by their FULL-season
    league mean mu_full[s]; the current first-k measurement demeans by the first-k
    causal mean mu_kt, so theta_hat is reported on the first-k mu_t scale. The
    league_shrunk and marcel benchmarks shrink toward the same first-k mu_kt."""
    model, m_rates, m_denoms, prior_pas = priors
    se2, su2 = phi_t

    # current first-k measurement (consumes rng)
    z_k, S_k = sample_measurement(cp["v_obs"], cp["d_obs"], B_BOOT, rng)

    # assemble measurements: priors first, current LAST
    z = [mt[0] for mt in model] + [z_k]
    mu = [mt[1] for mt in model] + [mu_kt]
    S = [mt[2] for mt in model] + [S_k]
    is_current = [False] * len(model) + [True]
    theta, V = cutpoint_posterior(np.array(z), np.array(mu), np.array(S),
                                  np.array(is_current), se2, su2)

    # leakage guard: conditioning rows = prior-season PAs + the first-k current PAs.
    # cutpoint_date is the k-th current PA's date (pas_sorted matches cp's split).
    first_k = pas_sorted.head(k).select("game_date", "season")
    cond = first_k if prior_pas is None else pl.concat(
        [prior_pas.select("game_date", "season"), first_k])
    cutpoint_date = pas_sorted["game_date"][k - 1]
    assert_causal(cond, cutpoint_date, t)

    # forward bootstrap over the player's causal value multiset (prior PAs + first-k)
    if prior_pas is None:
        ref_v, ref_d = cp["v_obs"], cp["d_obs"]
    else:
        ref_v = np.concatenate([prior_pas["value"].to_numpy(), cp["v_obs"]])
        ref_d = np.concatenate([prior_pas["denom"].to_numpy(), cp["d_obs"]])
    fc = forward_forecast(theta, V, cp["r_obs"], cp["w"], ref_v, ref_d,
                          m=len(cp["v_rest"]), B=B_FWD, rng=rng)

    # actual final line, and each benchmark as a point r_rest_hat blended to r_final
    r_final_actual = final_line_blend(cp["r_obs"], cp["D_obs"],
                                      cp["r_rest"], cp["D_rest"])[0]

    def blend(r_rest_hat: float) -> float:
        return final_line_blend(cp["r_obs"], cp["D_obs"], r_rest_hat, cp["D_rest"])[0]

    r_final_naive = cp["r_obs"]                       # blend(r_obs) == r_obs identically
    r_final_league = blend(league_shrunk(z_k, S_k, mu_kt, tau2_t))
    r_final_marcel = blend(marcel(m_rates, m_denoms, cp["r_obs"], cp["D_obs"], mu_kt))
    l2_rest = cutpoint_posterior(np.array([z_k]), np.array([mu_kt]),
                                 np.array([S_k]), np.array([True]), se2, su2)[0]
    r_final_l2 = blend(l2_rest)
    # savant_to_date == naive here: our BBE values ARE Savant est_woba, so Savant's
    # xwOBA through k equals cp["r_obs"]. Recorded as its own column for Task 10.
    r_final_savant = cp["r_obs"]

    row = {
        "batter": i, "season": t, "k": k, "w": cp["w"],
        "D_obs": cp["D_obs"], "D_rest": cp["D_rest"],
        "m": len(cp["v_rest"]), "n_prior": len(model),
        "theta_hat": theta, "V": V, "mu_t": mu_kt,
        "r_final_model": fc["center"],
        "q05": fc["q05"], "q10": fc["q10"], "q25": fc["q25"],
        "q75": fc["q75"], "q90": fc["q90"], "q95": fc["q95"],
        "r_final_actual": r_final_actual,
        "r_final_naive": r_final_naive,
        "r_final_league": r_final_league,
        "r_final_marcel": r_final_marcel,
        "r_final_l2": r_final_l2,
        "r_final_savant": r_final_savant,
    }
    return row, cond.height


def run_sweep(groups: dict, full: dict, mu_full: dict, mu_k: dict, tau2: dict,
              phi: dict, all_seasons: list[int], rng: np.random.Generator
              ) -> tuple[list, int, dict]:
    """Sweep target season -> cutpoint -> eligible player (all sorted, so the RNG
    stream is reproducible). Priors are memoized per (player, season) since they use
    no RNG and are independent of k."""
    rows: list[dict] = []
    n_cond = 0
    pair_counts: dict = {}
    prior_cache: dict = {}
    players = {t: sorted(b for (b, s) in groups if s == t) for t in all_seasons}
    for t in all_seasons:
        for k in CUTPOINTS:
            cnt = 0
            for i in players[t]:
                cp = cutpoint_split(groups[(i, t)], k, MIN_REMAINING)
                if cp is None:
                    continue
                key = (i, t)
                if key not in prior_cache:
                    prior_cache[key] = build_priors(i, t, all_seasons, groups,
                                                    full, mu_full)
                pas_sorted = groups[(i, t)].sort("game_date", maintain_order=True)
                row, ch = forecast_row(i, t, k, cp, pas_sorted, prior_cache[key],
                                       mu_k[(t, k)], tau2[t], phi[t], rng)
                rows.append(row); n_cond += ch; cnt += 1
            pair_counts[(t, k)] = cnt
        print(f"  season {t}: {sum(pair_counts[(t, kk)] for kk in CUTPOINTS)} forecasts")
    return rows, n_cond, pair_counts


def main():
    cfg = load_config()
    seasons = sorted(cfg.all_seasons)
    print("building PA frame from slim Statcast caches...")
    f = build_pa_frame(load_pitches(cfg, seasons))
    print(f"  {f.height:,} PAs over seasons {seasons}")
    rng = np.random.default_rng(cfg.seed)

    # causal league means: full-season (prior scale) and first-k (current scale)
    mu_full = {s: season_mu_causal(f, s, 10 ** 9) for s in seasons}
    mu_k = {(s, k): season_mu_causal(f, s, k) for s in seasons for k in CUTPOINTS}

    groups = f.partition_by(["batter", "season"], as_dict=True)
    print(f"  {len(groups):,} player-seasons")

    print("precomputing full-season measurements (one bootstrap each)...")
    full = precompute_full_measurements(groups, rng)
    tau2 = fit_tau2(full, seasons)
    phi = fit_loso_phi(full, mu_full, seasons)

    print("sweeping cutpoints...")
    rows, n_cond, pair_counts = run_sweep(groups, full, mu_full, mu_k, tau2, phi,
                                          seasons, rng)

    outdir = cfg.results_dir / "talent3"
    outdir.mkdir(parents=True, exist_ok=True)
    tbl = pl.DataFrame(rows)
    tbl.write_parquet(outdir / "forecast_table.parquet")

    print(f"\n  total forecast rows: {tbl.height:,}")
    print("  eligible pairs by (season, k):")
    header = "    season " + " ".join(f"k={k:<4}" for k in CUTPOINTS) + "  total"
    print(header)
    for t in seasons:
        cells = " ".join(f"{pair_counts[(t, k)]:>5}" for k in CUTPOINTS)
        tot = sum(pair_counts[(t, k)] for k in CUTPOINTS)
        print(f"    {t}    {cells}  {tot:>5}")

    digest = {"n_forecasts": tbl.height, "n_conditioning_rows": n_cond,
              "assert_causal": "passed", "seed": cfg.seed}
    (outdir / "leakage_digest.json").write_text(json.dumps(digest, indent=2))
    print(f"\n  LEAKAGE: assert_causal passed on all {tbl.height:,} forecasts "
          f"({n_cond:,} conditioning rows checked)")
    print(f"  wrote {outdir}/forecast_table.parquet + leakage_digest.json")


if __name__ == "__main__":
    main()
