"""Benchmark rest-of-season-rate forecasts for the talent3 evaluation (spec §8).
Pure functions; single_season_l2 and savant_to_date are assembled in the script
from existing tables."""
from __future__ import annotations


def naive(r_obs: float) -> float:
    """'He keeps hitting as he has.'"""
    return r_obs


def league_shrunk(z: float, s00: float, mu: float, tau2: float) -> float:
    """Phase-1 EB shrink of the first-k rate z toward the season mean mu (reproduces
    talent.eb_shrink). tau2 is the per-season between-player variance (fit once via
    talent.eb_fit on the season and passed in by the script). rel = tau2/(tau2+s00)."""
    rel = tau2 / (tau2 + s00)
    return mu + rel * (z - mu)


def marcel(prior_rates, prior_denoms, cur_rate: float, cur_denom: float,
           mu: float, regress_pa: float = 200.0, weights=(5, 4, 3)) -> float:
    """Marcel-style projection: recency-weighted mean of {prior seasons (most recent
    first), current-to-date}, then regressed toward the league mean mu by adding
    regress_pa league-average PAs. Denominator-weighted within each season."""
    num = cur_rate * cur_denom * weights[0]
    den = cur_denom * weights[0]
    for j, (r, d) in enumerate(zip(prior_rates, prior_denoms)):
        w = weights[min(j + 1, len(weights) - 1)]
        num += r * d * w
        den += d * w
    num += mu * regress_pa
    den += regress_pa
    return num / den
