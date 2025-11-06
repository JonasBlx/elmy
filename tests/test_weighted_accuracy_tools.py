from __future__ import annotations

import numpy as np

from src.utils.weighted_accuracy_and_tools import (
    decompose_y,
    reconstruct_y,
    weighted_accuracy_score,
)


def test_decompose_and_reconstruct_roundtrip() -> None:
    y = np.array([2.5, -1.0, 0.0, 3.0])
    direction, magnitude = decompose_y(y)
    reconstructed = reconstruct_y(direction, magnitude)
    np.testing.assert_allclose(reconstructed, np.array([2.5, -1.0, 0.0, 3.0]))


def test_weighted_accuracy_score_matches_manual_computation() -> None:
    y_true = np.array([2.0, -1.0, 3.0])
    y_pred = np.array([1.0, -0.5, -2.0])
    score = weighted_accuracy_score(y_true, y_pred)

    correct_direction = np.array([1, 1, 0], dtype=float)
    expected = np.sum(correct_direction * np.abs(y_true)) / np.sum(np.abs(y_true))
    assert score == expected
