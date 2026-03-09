"""Aggregate features: merchant fraud rates, device sharing, account-level stats."""

from __future__ import annotations

import polars as pl

from utils.logging import get_logger

log = get_logger(__name__)


def compute_aggregate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute aggregate-level features on the transaction DataFrame."""
    df = _merchant_fraud_rate(df)
    df = _device_sharing_degree(df)
    df = _devices_per_account(df)
    df = _new_merchants_windows(df)
    df = _amount_zscore(df)
    return df


def _merchant_fraud_rate(df: pl.DataFrame) -> pl.DataFrame:
    rates = (
        df.group_by("merchant_id")
        .agg(pl.col("is_fraud").mean().alias("merchant_fraud_rate"))
    )
    return df.join(rates, on="merchant_id", how="left")


def _device_sharing_degree(df: pl.DataFrame) -> pl.DataFrame:
    sharing = (
        df.group_by("device_id")
        .agg(pl.col("user_id").n_unique().alias("device_sharing_degree"))
    )
    return df.join(sharing, on="device_id", how="left")


def _devices_per_account(df: pl.DataFrame) -> pl.DataFrame:
    devs = (
        df.group_by("account_id")
        .agg(pl.col("device_id").n_unique().alias("devices_per_account"))
    )
    return df.join(devs, on="account_id", how="left")


def _new_merchants_windows(df: pl.DataFrame) -> pl.DataFrame:
    """Count of distinct merchants per user over 24h and 7d windows.

    Uses a sorted-order cumulative approach via row_number-based self-join.
    """
    df = df.with_row_index("_row_idx")
    new_24h = []
    new_7d = []

    ts_col = df["timestamp"].to_list()
    user_col = df["user_id"].to_list()
    merchant_col = df["merchant_id"].to_list()

    from datetime import timedelta

    merchant_history: dict[str, list[tuple]] = {}
    for i in range(len(df)):
        uid = user_col[i]
        mid = merchant_col[i]
        ts = ts_col[i]

        if uid not in merchant_history:
            merchant_history[uid] = []

        hist = merchant_history[uid]
        cutoff_24h = ts - timedelta(hours=24)
        cutoff_7d = ts - timedelta(days=7)

        merchants_24h = {m for t, m in hist if t >= cutoff_24h}
        merchants_7d = {m for t, m in hist if t >= cutoff_7d}

        new_24h.append(len(merchants_24h))
        new_7d.append(len(merchants_7d))

        hist.append((ts, mid))

    df = df.with_columns(
        pl.Series("new_merchants_24h", new_24h, dtype=pl.Float64),
        pl.Series("new_merchants_7d", new_7d, dtype=pl.Float64),
    ).drop("_row_idx")

    return df


def _amount_zscore(df: pl.DataFrame) -> pl.DataFrame:
    stats = df.group_by("user_id").agg(
        pl.col("amount").mean().alias("_user_amt_mean"),
        pl.col("amount").std().alias("_user_amt_std"),
    )
    df = df.join(stats, on="user_id", how="left")
    df = df.with_columns(
        ((pl.col("amount") - pl.col("_user_amt_mean")) / pl.col("_user_amt_std").clip(1e-6))
        .alias("amount_zscore")
    ).drop("_user_amt_mean", "_user_amt_std")
    return df
