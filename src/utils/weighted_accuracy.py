"""
Custom weighted accuracy metric used during the ENS Challenge Data competition.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import make_scorer

__all__ = ["weighted_accuracy", "weighted_accuracy_scorer"]

EPSILON: Final[float] = 1e-12


def weighted_accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, *, zero_division: float = 0.0
) -> float:
    """Compute the accuracy weighted by the absolute magnitude of the true values.

    Args:
        y_true: Ground-truth signed deviations.
        y_pred: Predicted signed deviations.
        zero_division: Value to return when the total weight is zero.

    Returns:
        A float between 0 and 1, where 1 signals perfect directional alignment.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("`y_true` and `y_pred` must share the same shape.")

    weights: NDArray[np.float_] = np.abs(y_true_arr)
    total_weight = np.sum(weights)
    if total_weight <= EPSILON:
        return float(zero_division)

    correct_direction = (np.sign(y_pred_arr) == np.sign(y_true_arr)).astype(float)
    score = float(np.sum(correct_direction * weights) / total_weight)
    return score


weighted_accuracy_scorer = make_scorer(weighted_accuracy, greater_is_better=True)
