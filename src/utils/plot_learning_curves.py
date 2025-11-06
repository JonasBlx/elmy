"""
Visualise model learning curves to diagnose bias-variance trade-offs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error

plt.style.use("seaborn-v0_8-darkgrid")


def plot_learning_curves(
    model: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    step: int = 24,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot RMSE curves for train and validation sets."""
    if step <= 0:
        raise ValueError("`step` must be a positive integer.")

    axis = ax or plt.gca()
    train_errors, val_errors = [], []
    train_sizes = range(step, len(X_train) + 1, step)

    for size in train_sizes:
        model.fit(X_train[:size], y_train[:size])
        train_pred = model.predict(X_train[:size])
        val_pred = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:size], train_pred, squared=False))
        val_errors.append(mean_squared_error(y_val, val_pred, squared=False))

    axis.plot(train_sizes, train_errors, label="Train RMSE", linewidth=2)
    axis.plot(train_sizes, val_errors, label="Validation RMSE", linewidth=2)
    axis.set_xlabel("Training samples")
    axis.set_ylabel("RMSE")
    axis.set_title("Learning Curves")
    axis.legend()
    return axis


def _load_dataset(
    path: Path, target_column: str, *, datetime_index: Optional[str] = None
) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    if datetime_index and datetime_index in frame.columns:
        frame = frame.set_index(datetime_index)
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[~frame.index.isna()]
    target = frame.pop(target_column)
    return frame, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning curves from CSV data.")
    parser.add_argument("--model", "-m", required=True, help="Path to a pickled sklearn model.")
    parser.add_argument("--train", "-t", required=True, help="Training CSV file.")
    parser.add_argument("--validation", "-v", required=True, help="Validation CSV file.")
    parser.add_argument("--target-column", "-y", required=True, help="Target column name.")
    parser.add_argument("--datetime-index", default=None, help="Optional datetime index column.")
    parser.add_argument("--step", type=int, default=24, help="Increment used to grow the training window.")
    parser.add_argument("--save", default="figures/learning_curve.png", help="Destination path for the PNG export.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model: RegressorMixin = joblib.load(args.model)
    X_train, y_train = _load_dataset(Path(args.train), args.target_column, datetime_index=args.datetime_index)
    X_val, y_val = _load_dataset(Path(args.validation), args.target_column, datetime_index=args.datetime_index)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_learning_curves(
        model,
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
        step=args.step,
        ax=ax,
    )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
