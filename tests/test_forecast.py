import numpy as np
from src.forecast import final_line_blend, forward_forecast


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


def test_blend_w_zero_ignores_nonfinite_r_rest():
    blend, w = final_line_blend(0.5, 1.0, float("inf"), 0.0)
    assert w == 0.0 and blend == 0.5   # 0*inf must NOT become nan


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
