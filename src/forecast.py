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
        out[f"q{round(lo*100):02d}"] = float(np.quantile(finals, lo))
        out[f"q{round(hi*100):02d}"] = float(np.quantile(finals, hi))
    return out


def _level_keys(levels):
    keys = []
    for lv in levels:
        lo = (1 - lv) / 2
        keys += [f"q{round(lo*100):02d}", f"q{round((1-lo)*100):02d}"]
    return keys
