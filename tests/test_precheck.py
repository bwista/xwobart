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
