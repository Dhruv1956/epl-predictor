"""Scoring rules and bookmaker-odds helpers.

Outcome order everywhere is [H, D, A] = [0, 1, 2]. `probs` has shape (n, 3) and each row
sums to 1; `y` holds the true outcome codes 0, 1 or 2.

    LL  = -(1/N) sum_i log p_i[y_i]                      (log loss, lower is better)
    BS  =  (1/N) sum_i sum_k (p_ik - 1[y_i = k])^2       (Brier score, lower is better)
    Acc =  (1/N) sum_i 1[argmax_k p_ik = y_i]            (ties resolve to home)
"""

from __future__ import annotations

import numpy as np


def log_loss(probs, y, eps=1e-15):
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs, y):
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs, y):
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    return float(np.mean(np.argmax(probs, axis=1) == y))


def scores(probs, y):
    return {
        "log_loss": log_loss(probs, y),
        "brier": brier_score(probs, y),
        "accuracy": accuracy(probs, y),
    }


def implied_probs(odds):
    """Bookmaker probabilities from decimal odds (n, 3), margin removed by normalisation.

    p_k = (1 / o_k) / sum_j (1 / o_j)
    """
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def overround(odds):
    """Bookmaker margin per match: sum_k 1/o_k - 1."""
    return (1.0 / np.asarray(odds, dtype=float)).sum(axis=1) - 1.0


def class_frequencies(y):
    """Share of home wins, draws and away wins."""
    y = np.asarray(y, dtype=int)
    return np.bincount(y, minlength=3) / len(y)
