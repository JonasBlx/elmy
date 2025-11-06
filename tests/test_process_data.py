from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.process_data import ProcessConfig, process_data, process_dataframe


def _build_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DELIVERY_START": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T02:00:00Z",
                "2024-01-01T03:00:00Z",
            ],
            "feature_a": [1.0, 2.0, np.nan, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
            "category": ["sunny", "cloudy", None, "sunny"],
            "remove_me": [0, 1, 2, 3],
        }
    )


def test_process_dataframe_scaling_and_imputation() -> None:
    df = _build_sample_dataframe()
    config = ProcessConfig(
        columns_to_remove=("remove_me",),
        scaler="standard",
    )

    processed = process_dataframe(df, config)

    assert processed.index.name == "DELIVERY_START"
    assert "remove_me" not in processed.columns
    assert processed["category"].isna().sum() == 0

    numeric = processed.select_dtypes(include="number")
    assert np.allclose(numeric.mean().values, 0, atol=1e-8)
    assert np.allclose(numeric.std(ddof=0).values, 1, atol=1e-8)


def test_process_data_wrapper_matches_dataframe_function() -> None:
    df = _build_sample_dataframe()
    wrapper_result = process_data(df, columns_to_remove=("remove_me",), lines_to_remove=(3,))
    config_result = process_dataframe(
        df,
        ProcessConfig(columns_to_remove=("remove_me",), lines_to_remove=(3,)),
    )
    pd.testing.assert_frame_equal(wrapper_result, config_result)
