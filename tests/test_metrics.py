"""Tests for src/metrics.py."""

import numpy as np
import pytest

from src import metrics as M


def test_uniform_reference_values():
    y = np.array([0, 1, 2, 0, 0, 1])
    probs = np.full((len(y), 3), 1 / 3)
    assert M.log_loss(probs, y) == pytest.approx(np.log(3))
    assert M.brier_score(probs, y) == pytest.approx(2 / 3)
    # argmax ties resolve to home, so accuracy equals the home-win share
    assert M.accuracy(probs, y) == pytest.approx(3 / 6)


def test_perfect_predictions():
    y = np.array([0, 1, 2])
    probs = np.eye(3)
    assert M.log_loss(probs, y) == pytest.approx(0.0, abs=1e-12)
    assert M.brier_score(probs, y) == pytest.approx(0.0)
    assert M.accuracy(probs, y) == 1.0


def test_log_loss_is_finite_when_true_outcome_gets_zero_probability():
    y = np.array([1])
    probs = np.array([[1.0, 0.0, 0.0]])
    assert np.isfinite(M.log_loss(probs, y))
    assert M.log_loss(probs, y) > 30  # large penalty, but not infinite


def test_known_log_loss_and_brier():
    y = np.array([0, 2])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert M.log_loss(probs, y) == pytest.approx(-np.log(0.5))
    per_match = (0.5**2) + (0.3**2) + (0.2**2)  # = 0.38 for both rows
    assert M.brier_score(probs, y) == pytest.approx(per_match)


def test_implied_probs_remove_the_margin():
    odds = np.array([[2.0, 3.5, 4.0], [1.5, 4.0, 7.0]])
    probs = M.implied_probs(odds)
    assert np.allclose(probs.sum(axis=1), 1.0)
    inv = 1 / odds[0]
    assert probs[0] == pytest.approx(inv / inv.sum())
    assert np.all(M.overround(odds) > 0)
    assert M.overround(odds)[0] == pytest.approx(inv.sum() - 1)


def test_class_frequencies():
    freqs = M.class_frequencies(np.array([0, 0, 1, 2, 0]))
    assert freqs == pytest.approx([0.6, 0.2, 0.2])
