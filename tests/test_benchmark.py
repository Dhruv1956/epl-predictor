import numpy as np
import pandas as pd

from src import benchmark as B


def test_rps_known_values():
    sure_home = np.array([[1.0, 0.0, 0.0]])
    assert B.rps(sure_home, [0]) == 0.0
    assert np.isclose(B.rps(sure_home, [1]), 0.5)   # called home, was a draw
    assert np.isclose(B.rps(sure_home, [2]), 1.0)   # called home, was away: two steps off
    uniform = np.full((1, 3), 1 / 3)
    assert np.isclose(B.rps(uniform, [0]), 0.5 * ((1 / 3) ** 2 + (2 / 3) ** 2))


def test_rps_prefers_a_near_miss_to_a_far_miss_unlike_brier():
    p_draw_ish = np.array([[0.1, 0.8, 0.1]])
    far = np.array([[0.8, 0.1, 0.1]])
    # outcome = away: forecast putting its mass on the draw is closer than one putting it on the home win
    assert B.rps(p_draw_ish, [2]) < B.rps(far, [2])


def test_book_probs_normalise_and_flag_missing():
    f = pd.DataFrame({"AvgCH": [2.0, np.nan], "AvgCD": [3.5, 3.0], "AvgCA": [4.0, 4.0]})
    p = B.book_probs(f, "AvgC")
    assert np.isclose(p[0].sum(), 1) and np.isnan(p[1]).all()
    assert p[0, 0] > p[0, 2]


def test_rps_per_match_matches_the_formula_by_hand():
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([0, 2])
    by_hand = [0.5 * ((0.5 - 1) ** 2 + (0.8 - 1) ** 2), 0.5 * ((0.2 - 0) ** 2 + (0.5 - 0) ** 2)]
    assert np.allclose(B.rps_per_match(p, y), by_hand)
