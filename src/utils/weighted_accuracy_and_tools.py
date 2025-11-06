"""
Helper utilities supporting the weighted accuracy metric.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def decompose_y(y: ArrayLike) -> Tuple[NDArray[np.int_], NDArray[np.float_]]:
    """Split signed deviations into a direction flag and magnitude."""
    y_arr = np.asarray(y, dtype=float)
    direction = (y_arr > 0).astype(int)
    magnitude = np.abs(y_arr)
    return direction, magnitude


def reconstruct_y(
    y_direction: ArrayLike, y_magnitude: ArrayLike
) -> NDArray[np.float_]:
    """Recombine direction and magnitude into a signed deviation."""
    direction_arr = np.asarray(y_direction, dtype=int)
    magnitude_arr = np.asarray(y_magnitude, dtype=float)

    if direction_arr.shape != magnitude_arr.shape:
        raise ValueError("`y_direction` and `y_magnitude` must have identical shapes.")

    sign_factors = np.where(direction_arr == 1, 1.0, -1.0)
    return magnitude_arr * sign_factors


def weighted_accuracy_score(
    y_true_reconstructed: ArrayLike, y_pred_simulated: ArrayLike
) -> float:
    """Compute the weighted accuracy score from reconstructed deviations."""
    y_true_arr = np.asarray(y_true_reconstructed, dtype=float)
    y_pred_arr = np.asarray(y_pred_simulated, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("`y_true_reconstructed` and `y_pred_simulated` must match in shape.")

    correct_direction = ((y_true_arr * y_pred_arr) > 0).astype(float)
    weights = np.abs(y_true_arr)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0

    return float(np.sum(correct_direction * weights) / total_weight)
