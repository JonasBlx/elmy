from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from src.utils.plot_learning_curves import plot_learning_curves


def test_plot_learning_curves_returns_axis() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(120, 3))
    coefs = np.array([0.5, -0.2, 0.1])
    y_train = X_train @ coefs + rng.normal(scale=0.1, size=120)

    X_val = rng.normal(size=(40, 3))
    y_val = X_val @ coefs + rng.normal(scale=0.1, size=40)

    model = LinearRegression()
    fig, ax = plt.subplots()
    returned_ax = plot_learning_curves(model, X_train, y_train, X_val, y_val, step=20, ax=ax)

    assert returned_ax is ax
    assert len(ax.lines) == 2
    plt.close(fig)
