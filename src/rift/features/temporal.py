"""Temporal feature utilities for Rift."""

from typing import Optional

import polars as pl


def time_since_last(df: pl.DataFrame, group_col: str, ts_col: str) -> pl.Expr:
    """Time since last transaction in seconds."""
    return pl.col(ts_col).diff().over(group_col).dt.total_seconds()


def hour_of_day(ts_col: str) -> pl.Expr:
    """Extract hour of day from timestamp."""
    return pl.col(ts_col).dt.hour()


def day_of_week(ts_col: str) -> pl.Expr:
    """Extract day of week from timestamp."""
    return pl.col(ts_col).dt.weekday()
