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
