import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from src import league2 as L


def _frame(n=500, seed=0, m_true=-0.5):
    """Two leagues of 6 teams; teams 0-1 move from league 1 to league 0 in the second half."""
    rng = np.random.default_rng(seed)
    rows = []
    date0 = pd.Timestamp("2019-08-10")
    strength = rng.normal(0, 0.3, 12)
    for i in range(n):
        d = date0 + pd.Timedelta(days=int(i * 700 / n))
        late = d > pd.Timestamp("2020-08-01")
        if rng.random() < 0.5:
            lg, pool = 0, ([0, 1, 2, 3, 4, 5] if late else [2, 3, 4, 5, 6, 7])
        else:
            lg, pool = 1, ([6, 7, 8, 9, 10, 11] if late else [0, 1, 8, 9, 10, 11])
        h, a = rng.choice(pool, 2, replace=False)
        sh, sa = strength[h] + (m_true if lg == 1 else 0), strength[a] + (m_true if lg == 1 else 0)
        rows.append((d, h, a, rng.poisson(np.exp(0.2 + 0.2 + sh - sa * 0)), rng.poisson(np.exp(0.2 + sa - sh * 0)), lg))
    f = pd.DataFrame(rows, columns=["Date", "h2", "a2", "FTHG", "FTAG", "league"])
    f["y"] = np.where(f.FTHG > f.FTAG, 0, np.where(f.FTHG == f.FTAG, 1, 2))
    f["HomeTeam"], f["AwayTeam"] = f.h2.astype(str), f.a2.astype(str)
    return f.sort_values("Date").reset_index(drop=True)


def test_periods_never_cross_a_season_and_cells_have_one_league():
    f = _frame()
    pk = L.period_key(f["Date"])
    assert (pk // 100 == L.season_year(f["Date"])).all()
    c = L.build_cells(f)
    assert c["n"] > 0 and set(np.unique(c["flag"])) <= {0.0, 1.0}


def test_previous_cell_links_follow_time_within_a_team():
    c = L.build_cells(_frame())
    has = c["prev"] >= 0
    assert (c["team"][has] == c["team"][c["prev"][has]]).all()
    assert (c["keys"][has, 0] > c["keys"][c["prev"][has], 0]).all()


def test_gradient_matches_finite_differences():
    f = _frame(n=300)
    c = L.build_cells(f)
    rng = np.random.default_rng(3)
    th = np.concatenate([[0.1, 0.2, 0.05, -0.1, -0.3, -0.2], rng.normal(0, 0.2, 2 * c["n"])])
    x, y, w, l = f.FTHG.values.astype(float), f.FTAG.values.astype(float), np.ones(len(f)), f.league.values.astype(float)
    _, g = L.loglik_and_grad(th, c, x, y, w, l, 5.0, 12.0)
    num = approx_fprime(th, lambda t: L.loglik_and_grad(t, c, x, y, w, l, 5.0, 12.0)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=2e-3)


def test_movers_identify_a_weaker_second_league():
    f = _frame(n=2500, m_true=-0.6, seed=2)
    c = L.build_cells(f)
    x, y, w, l = f.FTHG.values.astype(float), f.FTAG.values.astype(float), np.ones(len(f)), f.league.values.astype(float)
    th = L.fit_cells(c, x, y, w, l, 5.0, 20.0)
    assert th[4] < -0.05 or th[5] > 0.05  # the second league is estimated weaker (attack down or defence down)


def test_forecast_does_not_use_later_results():
    f = _frame(n=400, seed=5)
    cut = pd.Timestamp("2020-12-01")
    a, _ = L.walk_forward_two(f, "2020-10-01", cut, 5.0, 20.0)
    f2 = f.copy()
    late = f2["Date"] >= cut
    f2.loc[late, ["FTHG", "FTAG"]] = np.random.default_rng(1).poisson(3, (late.sum(), 2))
    b, _ = L.walk_forward_two(f2, "2020-10-01", cut, 5.0, 20.0)
    assert np.allclose(a.values, b.values)
