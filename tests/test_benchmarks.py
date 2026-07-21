import numpy as np
from src.benchmarks import naive, marcel, league_shrunk

def test_naive_is_observed_rate():
    assert naive(0.351) == 0.351

def test_league_shrunk_matches_phase1_formula():
    # rel = tau2/(tau2+s00); theta = mu + rel*(z-mu). Reproduces talent.eb_shrink.
    z, s00, mu, tau2 = 0.400, 0.050 ** 2, 0.315, 0.031 ** 2
    rel = tau2 / (tau2 + s00)
    assert abs(league_shrunk(z, s00, mu, tau2) - (mu + rel * (z - mu))) < 1e-12
    # huge measurement variance -> shrinks fully to mu
    assert abs(league_shrunk(0.9, 10.0, 0.315, tau2) - 0.315) < 1e-3

def test_marcel_weights_and_regression():
    # one prior season rate .340 over 400 denom; current .360 over 100 denom;
    # league mu .315; regress with 200 league PA. Weighted-mean then shrink.
    est = marcel(prior_rates=[0.340], prior_denoms=[400.0],
                 cur_rate=0.360, cur_denom=100.0, mu=0.315,
                 regress_pa=200.0, weights=(5, 4, 3))
    # manual: recency weight prior=5*400, current gets weight 5*100? -> see impl doc
    assert 0.315 < est < 0.360           # between league and current, sensible
    # regression pulls toward mu: with regress_pa large, est -> mu
    est_heavy = marcel([0.340], [400.0], 0.360, 100.0, 0.315,
                       regress_pa=100_000.0)
    assert abs(est_heavy - 0.315) < 0.01
