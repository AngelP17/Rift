"""Temporal window utilities for graph construction."""

from datetime import timedelta
from typing import Optional

import polars as pl


def rolling_window(
    df: pl.DataFrame,
    ts_col: str,
    window_days: int = 7,
) -> pl.DataFrame:
    """Filter to transactions within rolling window of most recent."""
    df = df.with_columns(pl.col(ts_col).str.to_datetime())
    max_ts = df[ts_col].max()
    cutoff = max_ts - timedelta(days=window_days)
    return df.filter(pl.col(ts_col) >= cutoff)
