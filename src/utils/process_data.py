"""
Dataframe preprocessing utilities tailored for the Elmy forecasting project.

The module retains the original behaviour of :func:`process_data` while
providing a modern, type-annotated workflow and a small CLI helpers. Use the
dataclass-powered :func:`process_dataframe` entry point whenever you need fine
grained control over the cleaning steps.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Hashable, Literal

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

LOGGER = logging.getLogger(__name__)

ScalerName = Literal["standard", "minmax"]
ScalerSpec = Union[ScalerName, TransformerMixin]

DEFAULT_INDEX_COLUMN = "DELIVERY_START"


@dataclass(frozen=True)
class ProcessConfig:
    """Configuration describing how a dataframe should be prepared."""

    columns_to_remove: Sequence[str] = ()
    lines_to_remove: Sequence[Hashable] = ()
    scaler: Optional[ScalerSpec] = None
    index_column: Optional[str] = DEFAULT_INDEX_COLUMN
    utc_index: bool = True
    fill_missing: bool = True


def process_dataframe(dataframe: pd.DataFrame, config: ProcessConfig) -> pd.DataFrame:
    """Clean and optionally scale a dataframe using :class:`ProcessConfig`.

    Args:
        dataframe: Raw input dataframe.
        config: Processing instructions.

    Returns:
        A new dataframe with rows/columns removed, missing values imputed and
        the requested scaler applied to numeric columns.
    """
    df = dataframe.copy(deep=True)

    if config.lines_to_remove:
        LOGGER.debug("Dropping %d rows", len(config.lines_to_remove))
        df = df.drop(index=list(config.lines_to_remove), errors="ignore")

    if config.index_column and config.index_column in df.columns:
        LOGGER.debug("Setting %s as index", config.index_column)
        df = df.set_index(config.index_column, drop=True)
        if config.utc_index:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            drop_mask = df.index.isna()
            if drop_mask.any():
                LOGGER.warning(
                    "Removed %d rows with invalid datetime index values", drop_mask.sum()
                )
                df = df[~drop_mask]

    if config.columns_to_remove:
        LOGGER.debug("Dropping %d columns", len(config.columns_to_remove))
        df = df.drop(columns=list(config.columns_to_remove), errors="ignore")

    if config.fill_missing:
        df = _fill_missing_values(df)

    scaler = _resolve_scaler(config.scaler)
    if scaler is not None:
        df = _scale_numeric_columns(df, scaler)

    return df


def process_data(
    dataframe: pd.DataFrame,
    columns_to_remove: Optional[Iterable[str]] = None,
    lines_to_remove: Optional[Iterable[Hashable]] = None,
    scaler: Optional[ScalerSpec] = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper around :func:`process_dataframe`."""
    config = ProcessConfig(
        columns_to_remove=tuple(columns_to_remove or ()),
        lines_to_remove=tuple(lines_to_remove or ()),
        scaler=scaler,
    )
    return process_dataframe(dataframe, config)


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with per-column strategies."""
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    non_numeric_columns = df.columns.difference(numeric_columns)
    for column in non_numeric_columns:
        if not df[column].isna().any():
            continue
        mode = df[column].mode(dropna=True)
        if not mode.empty:
            df[column] = df[column].fillna(mode.iloc[0])
        else:
            df[column] = df[column].fillna(method="ffill").fillna(method="bfill")
    return df


def _resolve_scaler(spec: Optional[ScalerSpec]) -> Optional[TransformerMixin]:
    if spec is None:
        return None
    if isinstance(spec, TransformerMixin):
        return spec
    spec_lc = spec.lower()
    if spec_lc == "standard":
        return StandardScaler()
    if spec_lc == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler specifier: {spec}")


def _scale_numeric_columns(df: pd.DataFrame, scaler: TransformerMixin) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        LOGGER.info("No numeric columns available for scaling; skipping scaler step.")
        return df
    scaled = scaler.fit_transform(df[numeric_columns])
    df.loc[:, numeric_columns] = scaled
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw Elmy energy datasets and write a cleaned CSV."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output", "-o", required=True, help="Destination path for the processed CSV."
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        help="Optional columns to drop during preprocessing.",
    )
    parser.add_argument(
        "--drop-rows",
        nargs="*",
        default=None,
        help="Optional row labels to drop before setting the index.",
    )
    parser.add_argument(
        "--scaler",
        choices=("standard", "minmax"),
        default=None,
        help="Scaler to apply to numeric columns.",
    )
    parser.add_argument(
        "--index-column",
        default=DEFAULT_INDEX_COLUMN,
        help="Column name to promote to the datetime index.",
    )
    parser.add_argument(
        "--no-utc",
        action="store_true",
        help="Skip UTC conversion when parsing the timestamp index.",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Skip missing-value imputation.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    LOGGER.info("Loading raw data from %s", input_path)
    df = pd.read_csv(input_path)

    config = ProcessConfig(
        columns_to_remove=tuple(args.drop_columns or ()),
        lines_to_remove=tuple(args.drop_rows or ()),
        scaler=args.scaler,
        index_column=args.index_column,
        utc_index=not args.no_utc,
        fill_missing=not args.no_fill_missing,
    )
    processed = process_dataframe(df, config)

    LOGGER.info("Writing cleaned data to %s", output_path)
    processed.to_csv(output_path)


if __name__ == "__main__":
    main()
