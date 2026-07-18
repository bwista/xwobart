import numpy as np
import polars as pl

from src.rollup import expected_values, linear_weights


def test_linear_weights_empirical_means():
    bbe = pl.DataFrame({
        "outcome_class": [0, 0, 0, 1, 1, 2, 4],
        "woba_value":    [0.0, 0.0, 0.9, 0.9, 0.9, 1.25, 2.0],  # one ROE in the out class
    })
    w, warnings = linear_weights(bbe)
    assert w.shape == (5,)
    assert abs(w[0] - 0.3) < 1e-9          # empirical mean, NOT forced 0 (spec §8.1)
    assert w[1] == 0.9 and w[2] == 1.25 and w[3] == 0.0 and w[4] == 2.0
    assert any("w[0]" in msg for msg in warnings)   # 0.3 is outside the sanity range
    assert any("w[3]" in msg for msg in warnings)   # empty triple class -> 0.0 flagged


def test_expected_values_dot():
    w = np.array([0.0, 1.0, 0.0, 0.0, 2.0])
    p = np.array([  # (S=2, K=5, n=2)
        [[0.5, 0.0], [0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.5], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    ev = expected_values(p, w)
    assert ev.shape == (2, 2)
    assert np.allclose(ev, [[0.5, 2.0], [0.0, 0.5]])
    assert ev.dtype == np.float32
