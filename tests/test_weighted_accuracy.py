from __future__ import annotations

import numpy as np
import pytest

from src.utils.weighted_accuracy import weighted_accuracy


def test_weighted_accuracy_perfect_alignment() -> None:
    y_true = np.array([1.0, -2.0, 3.0])
    y_pred = np.array([2.0, -1.0, 4.0])
    assert weighted_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_weighted_accuracy_zero_weights_returns_zero_division_value() -> None:
    y_true = np.zeros(4)
    y_pred = np.ones(4)
    assert weighted_accuracy(y_true, y_pred, zero_division=0.75) == pytest.approx(0.75)


def test_weighted_accuracy_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        weighted_accuracy(np.array([1, 2]), np.array([1, 2, 3]))
