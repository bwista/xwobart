"""Stage 3, run 2 of 2: spray-MARGINALIZED per-event values, from the pickled trees.
Run from repo root AFTER `run_v0.py --stage C --variant spray`:
    .venv/bin/python scripts/marginalize_spray.py [-M 9]

Why this exists as a separate process. The plan folded marginalization into the fit
because the fitted trees live on the in-memory model object and "cannot be recovered from
idata.nc". That is true of idata.nc -- but `run_v0.py` now pickles the trees correctly
(they are a multiprocessing Manager ListProxy, so they must be materialized with list()
before pickling; pickling the proxy writes a dead ~200-byte token). With a real pickle the
60-minute job splits into two ~30-minute halves, and the ELPD verdict lands after the first.

The marginalized value replaces v(x_e) with its league average over spray given
(EV x LA x stand) -- design risk 2: conditioning a player rollup on per-ball direction
credits spray LUCK. No refit; just M extra prediction passes.
Plan: docs/superpowers/plans/2026-07-19-xwobart-phase2-stage23-spray-surface.md"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src import data, prep, rollup
from src.config import load_config
from src.model import _softmax
from src.prep import K


def thinned_draw_count(chains: int, draws: int, max_total: int) -> int:
    """Mirror src.model.thin exactly, so S matches the fit's own prediction passes."""
    per_chain = max(1, max_total // chains)
    if draws <= per_chain:
        return chains * draws
    step = int(math.ceil(draws / per_chain))
    return len(range(0, draws, step)) * chains


def predict_ev_mean(trees, X: np.ndarray, w: np.ndarray, S: int, seed: int,
                    chunk: int) -> np.ndarray:
    """Posterior-mean expected wOBA value per row, from stored trees.

    Mirrors model.predict_and_reduce's chunked path (and its (S,n,K)->(S,K,n) transpose,
    which is the pymc-bart 0.12 convention), but keeps only the draw mean -- the A/B needs
    the mean, not the spread, and the full (S, n) stack at 363,595 rows is 727 MB."""
    from pymc_bart.utils import _sample_posterior

    out = np.empty(X.shape[0], dtype=np.float64)
    for lo in range(0, X.shape[0], chunk):
        hi = min(lo + chunk, X.shape[0])
        rng = np.random.default_rng(seed)
        mu = np.asarray(_sample_posterior(trees, X=X[lo:hi], rng=rng, size=S,
                                          shape=K)).transpose(0, 2, 1)   # (S, K, c)
        out[lo:hi] = rollup.expected_values(_softmax(mu, axis=1), w).mean(axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", "--marginalize-spray", type=int, default=9, metavar="M",
                    help="equal-mass league spray quantiles per (EV, LA, stand) cell")
    args = ap.parse_args()
    M = args.marginalize_spray

    t0 = time.perf_counter()
    cfg = load_config()
    stage_dir = cfg.results_dir / "stage_C_spray"
    features = prep.VARIANT_FEATURES["spray"]
    si = features.index("spray_pull")

    print("[1/4] loading the pickled trees + persisted keys")
    with open(stage_dir / "all_trees.pkl", "rb") as f:
        trees = pickle.load(f)
    metrics = json.loads((stage_dir / "metrics.json").read_text())
    st = metrics["stage_settings"]
    S = thinned_draw_count(st["chains"], st["draws"], cfg.thin_draws)
    keys = {t: pl.read_parquet(stage_dir / f"ev_draws_keys_{t}.parquet")
            for t in ("train", "holdout")}
    print(f"  {len(trees)} tree draws on disk; predicting with S={S}; "
          f"rows train {keys['train'].height:,} holdout {keys['holdout'].height:,}")

    # Linear weights are recomputed from the caches rather than read from metrics.json,
    # whose copy is rounded to 4 dp. Deterministic and identical to the fit's own w.
    print("[2/4] recomputing linear weights from the training BBE")
    sprint = data.fetch_sprint_speed(cfg)
    bbe_train, _ = prep.filter_bbe(data.load_seasons(cfg, cfg.train_seasons))
    bbe_train, _ = data.merge_sprint_speed(bbe_train, sprint)
    bbe_train = prep.add_outcome_class(bbe_train)
    w, w_warn = rollup.linear_weights(bbe_train)
    stored = metrics["linear_weights"]
    # metrics.json stores np.round(w, 4), so the rounding error is up to exactly 5e-5;
    # 1e-4 clears that boundary while still catching any real mismatch (class weights
    # are ~0.9 apart).
    assert all(abs(w[i] - stored[c]) < 1e-4 for i, c in enumerate(prep.CLASS_NAMES)), \
        f"recomputed weights {np.round(w, 4).tolist()} != the fit's {stored}"
    print(f"  w {np.round(w, 4).tolist()} (matches the fit) {w_warn or ''}")

    # Quantile table from the TRAINING rows only. At Stage C predict_cap is None, so the
    # train key parquet IS bbe_train row-for-row -- same table run_v0 would have built.
    print(f"[3/4] {M} equal-mass spray quantiles per (EV, LA, stand) cell")
    qs = np.linspace(0.5 / M, 1 - 0.5 / M, M)
    qt = (keys["train"].with_columns(
              _ev=(pl.col("launch_speed") // prep.SPRAY_EV_BIN).cast(pl.Int32),
              _la=(pl.col("launch_angle") // prep.SPRAY_LA_BIN).cast(pl.Int32))
          .group_by("_ev", "_la", "stand_R")
          .agg([pl.col("spray_pull").quantile(q).alias(f"q{i}") for i, q in enumerate(qs)],
               _n=pl.len())
          .filter(pl.col("_n") >= prep.SPRAY_MIN_CELL).drop("_n")
          .sort("_ev", "_la", "stand_R"))   # polars group_by is not order-stable
    print(f"  {qt.height} dense cells")

    print("[4/4] marginalizing")
    marg, sparse_rate = {}, {}
    for tag, pt in keys.items():
        j = (pt.with_columns(
                 _ev=(pl.col("launch_speed") // prep.SPRAY_EV_BIN).cast(pl.Int32),
                 _la=(pl.col("launch_angle") // prep.SPRAY_LA_BIN).cast(pl.Int32))
               .join(qt, on=["_ev", "_la", "stand_R"], how="left").sort("row"))
        assert j.height == pt.height, "marginalization join changed row count"
        assert j["row"].to_numpy().tolist() == list(range(pt.height)), \
            "row order not restored — the .npy alignment contract would break"
        Q = j.select([f"q{i}" for i in range(M)]).to_numpy()          # (n, M)
        base = j.select(features).to_numpy().astype(np.float64)       # (n, 5)
        # Capture sparsity BEFORE the fill: after np.where, Q is all-finite and the rate
        # would read 0.0, silently killing the no-op diagnostic.
        sparse_rate[tag] = float(np.mean(~np.isfinite(Q)))
        Q = np.where(np.isfinite(Q), Q, base[:, si:si + 1])   # sparse cells keep observed
        acc = np.zeros(base.shape[0])
        for m in range(M):
            Xm = base.copy()
            Xm[:, si] = Q[:, m]
            acc += predict_ev_mean(trees, Xm, w, S, cfg.seed, cfg.chunk_size)
            print(f"  {tag} pass {m + 1}/{M} ({(time.perf_counter() - t0) / 60:.1f} min)")
        marg[tag] = acc / M
        np.save(stage_dir / f"ev_marginalized_{tag}.npy", marg[tag].astype(np.float32))

    cond_train = np.load(stage_dir / "ev_draws_train.npy").mean(axis=0)
    metrics["marginalize_spray"] = {
        "M": M,
        "sparse_cell_rate": sparse_rate,
        "mean_abs_shift_train": float(np.abs(marg["train"] - cond_train).mean()),
        "from_pickled_trees": True,
        "runtime_s": time.perf_counter() - t0,
    }
    (stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nmarginalized: {metrics['marginalize_spray']}")
    if metrics["marginalize_spray"]["mean_abs_shift_train"] == 0.0:
        raise SystemExit("STOP: shift is exactly 0 — marginalization silently no-oped")
    print(f"wrote ev_marginalized_*.npy in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
