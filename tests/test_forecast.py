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
