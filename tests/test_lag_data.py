from __future__ import annotations

import pandas as pd

from src.utils.lag_data import LagConfig, generate_lagged_dataset


def test_generate_lagged_dataset_creates_expected_columns() -> None:
    features = pd.DataFrame(
        {
            "sensor_a": [1, 2, 3, 4, 5],
            "sensor_b": [5, 4, 3, 2, 1],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC"),
    )
    target = pd.Series([10, 20, 30, 40, 50], index=features.index, name="target")

    lagged, aligned_target = generate_lagged_dataset(features, target, LagConfig(n_lags=2))

    assert lagged.shape[0] == 3  # two initial rows dropped due to lagging
    assert aligned_target.index.equals(lagged.index)
    expected_columns = {
        "sensor_a_lag_0",
        "sensor_a_lag_1",
        "sensor_a_lag_2",
        "sensor_b_lag_0",
        "sensor_b_lag_1",
        "sensor_b_lag_2",
    }
    assert set(lagged.columns) == expected_columns


def test_generate_lagged_dataset_keep_na() -> None:
    features = pd.DataFrame({"sensor": [1, 2, 3]})
    target = pd.Series([1, 0, 1])

    lagged, _ = generate_lagged_dataset(features, target, LagConfig(n_lags=1, drop_missing=False))
    assert lagged.isna().sum().sum() == 2  # first row contains NaNs for lagged values
