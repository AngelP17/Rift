"""Aggregate feature computations for Rift."""

import polars as pl

from rift.data.schemas import AMOUNT, MERCHANT_ID, TIMESTAMP, USER_ID


def rolling_aggregates(
    df: pl.DataFrame,
    group_col: str = USER_ID,
    ts_col: str = TIMESTAMP,
    amount_col: str = AMOUNT,
) -> pl.DataFrame:
    """Compute rolling aggregates per group."""
    df = df.with_columns(pl.col(ts_col).str.to_datetime())
    df = df.sort(ts_col)

    # Use map_groups for rolling - simplified version
    return df.with_columns([
        pl.col(amount_col).cum_sum().over(group_col).alias("cum_spend"),
        pl.col(amount_col).cum_count().over(group_col).alias("cum_count"),
    ])
