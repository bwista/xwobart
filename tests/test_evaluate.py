import numpy as np

from src.evaluate import brier_score, elpd_metrics, reliability_curve


def test_reliability_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.uniform(size=20_000) < p).astype(float)
    curve = reliability_curve(p, y, n_bins=10)
    assert curve["ece"] < 0.02
    assert len(curve["conf"]) <= 10 and len(curve["conf"]) == len(curve["acc"])
    assert sum(curve["count"]) == 20_000


def test_reliability_degenerate_probs_no_crash():
    # Triple-like: nearly all mass at ~0 -> duplicate quantile edges must collapse
    p = np.concatenate([np.full(9_990, 0.001), np.full(10, 0.4)])
    y = np.zeros(10_000)
    curve = reliability_curve(p, y, n_bins=10)
    assert np.isfinite(curve["ece"])
    assert sum(curve["count"]) == 10_000


def test_brier():
    p = np.array([1.0, 0.0, 0.5])
    y = np.array([1.0, 0.0, 1.0])
    assert abs(brier_score(p, y) - (0.25 / 3)) < 1e-9


def test_elpd_known_values():
    # Two events, two draws, K=2. Event 0: p[y]=0.5 both draws. Event 1: 0.25 / 0.75.
    lppd0 = np.log(0.5)
    lppd1 = np.log((0.25 + 0.75) / 2)
    lppd_i = np.array([lppd0, lppd1])
    meanlog_i = np.array([np.log(0.5), (np.log(0.25) + np.log(0.75)) / 2])
    m = elpd_metrics(lppd_i, meanlog_i)
    assert abs(m["elpd_lppd"] - (lppd0 + lppd1)) < 1e-12
    assert abs(m["elpd_se"] - np.sqrt(2 * np.var(lppd_i))) < 1e-12
    assert m["mean_log_lik_sum"] < m["elpd_lppd"]   # Jensen: mean-of-log <= log-of-mean
    assert m["n_events"] == 2


from src.evaluate import binned_residuals, undercorrection


def test_binned_residuals_recovers_offset():
    ev = np.linspace(60, 110, 1000)
    model_val = np.full(1000, 0.40)
    public_val = np.where(ev > 100, 0.30, 0.40)   # model reads +0.10 high above EV 100
    table = binned_residuals(model_val, public_val, ev, lo=55, hi=115, width=5)
    high = [r for r in table if r["bin_lo"] >= 100]
    low = [r for r in table if r["bin_hi"] <= 100]
    assert all(abs(r["mean_residual"] - 0.10) < 1e-9 for r in high)
    assert all(abs(r["mean_residual"]) < 1e-9 for r in low)


def test_undercorrection_directional():
    rng = np.random.default_rng(1)
    sprint = rng.uniform(24, 30, 5000)
    # Truth: fast players beat out grounders. Public ignores speed; model corrects.
    actual = 0.05 * (sprint - 27) + rng.normal(0, 0.02, 5000)
    public_pred = np.zeros(5000)
    model_pred = 0.05 * (sprint - 27)
    m = undercorrection(actual, model_pred, public_pred, sprint)
    assert m["public_residual_sprint_corr"] > 0.5
    assert abs(m["model_residual_sprint_corr"]) < 0.1


from src.evaluate import contact_grids


def test_contact_grids():
    s, X_g, X_b = contact_grids((23.0, 31.0, 5))
    assert s.tolist() == [23.0, 25.0, 27.0, 29.0, 31.0]
    assert X_g.shape == (5, 3) and X_b.shape == (5, 3)
    assert np.all(X_g[:, 0] == 85.0) and np.all(X_g[:, 1] == -10.0)
    assert np.all(X_b[:, 0] == 103.0) and np.all(X_b[:, 1] == 28.0)
    assert np.all(X_g[:, 2] == s)


from src.evaluate import contact_grids


def test_contact_grids_v0_unchanged():
    s, g, b = contact_grids((23.0, 31.0, 5))
    assert s.shape == (5,) and g.shape == (5, 3) and b.shape == (5, 3)
    assert g[0].tolist() == [85.0, -10.0, 23.0]
    assert b[0].tolist() == [103.0, 28.0, 23.0]


def test_contact_grids_spray_variant_adds_pulled_and_oppo_grounders():
    s, grids = contact_grids((23.0, 31.0, 5), variant="spray")
    assert set(grids) == {"grounder_pull", "grounder_oppo", "barrel_pull"}
    for name, X in grids.items():
        assert X.shape == (5, 5)
        assert X[0, 3] == 1.0                      # stand_R: all grids are RHB
        assert X[0, 4] == 23.0                     # sprint speed is the last column
    assert grids["grounder_pull"][0].tolist() == [85.0, -10.0, 20.0, 1.0, 23.0]
    assert grids["grounder_oppo"][0].tolist() == [85.0, -10.0, -20.0, 1.0, 23.0]
    assert grids["barrel_pull"][0].tolist() == [103.0, 28.0, 20.0, 1.0, 23.0]


def test_la_spray_grid_is_row_major_over_la_then_spray():
    from src.evaluate import la_spray_grid

    la_ax, sp_ax, X = la_spray_grid(la=(0.0, 30.0, 4), spray=(-20.0, 20.0, 3))
    assert la_ax.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert sp_ax.tolist() == [-20.0, 0.0, 20.0]
    assert X.shape == (12, 5)
    # row-major: LA is the slow axis, spray the fast one -- the reshape in the figure
    # code is X[:, k].reshape(len(la_ax), len(sp_ax)) and depends on exactly this
    assert X[:3, 1].tolist() == [0.0, 0.0, 0.0]
    assert X[:3, 2].tolist() == [-20.0, 0.0, 20.0]
    assert X[3, 1] == 10.0
    assert (X[:, 0] == 103.0).all() and (X[:, 3] == 1.0).all()
