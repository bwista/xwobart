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
                barrel: np.ndarray, B: int, rng: np.random.Generator,
                pull: np.ndarray | None = None) -> np.ndarray:
    """Measurement covariance of the observed triple (or quadruple, with pull)
    for ONE player-season, by resampling their PAs with replacement. ev/barrel/
    pull are NaN on non-BBE rows; replicates are computed with NaN-aware means,
    and replicates with zero tracked BBE get NaN peripherals. Entries touching
    a peripheral are NaN when fewer than B/2 replicates were valid (caller
    falls back to 1-D). The xwOBA variance is floored at FLOOR_SD_PER_PA^2/n so
    degenerate tiny samples cannot claim certainty (Phase-1 NOTES limitation
    3). pull defaults to None, giving the original 3x3 (xwoba, ev, barrel)
    behavior bit-for-bit (same single rng.integers draw, same arithmetic);
    when given, pull is resampled with that SAME idx (no extra rng call) and
    gets its own valid-count, since it can be NaN (missing hit coords) on BBE
    where ev/barrel are finite -- the return is then 4x4 over (xwoba, ev,
    barrel, pull)."""
    n = len(value)
    idx = rng.integers(0, n, size=(B, n))
    v, d, e, b = value[idx], denom[idx], ev[idx], barrel[idx]
    den = d.sum(axis=1)
    xw = np.where(den > 0, v.sum(axis=1) / np.maximum(den, 1e-12), np.nan)
    ecnt = np.isfinite(e).sum(axis=1)
    ev_rep = np.where(ecnt > 0, np.nansum(e, axis=1) / np.maximum(ecnt, 1), np.nan)
    br_rep = np.where(ecnt > 0, np.nansum(b, axis=1) / np.maximum(ecnt, 1), np.nan)
    reps = [xw, ev_rep, br_rep]
    if pull is not None:
        p = pull[idx]
        pcnt = np.isfinite(p).sum(axis=1)
        pull_rep = np.where(pcnt > 0, np.nansum(p, axis=1) / np.maximum(pcnt, 1), np.nan)
        reps.append(pull_rep)

    D = len(reps)
    S = np.full((D, D), np.nan)
    ok_x = np.isfinite(xw)
    if ok_x.sum() >= B // 2:
        S[0, 0] = xw[ok_x].var(ddof=1)
    ok_all = ok_x
    for r in reps[1:]:
        ok_all = ok_all & np.isfinite(r)
    if ok_all.sum() >= B // 2:
        S = np.cov(np.stack([r[ok_all] for r in reps]), ddof=1)
    if np.isfinite(S[0, 0]):
        S[0, 0] = max(S[0, 0], FLOOR_SD_PER_PA ** 2 / n)   # raises an eigenvalue: stays PSD
    for k in range(1, D):
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
