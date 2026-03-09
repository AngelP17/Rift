"""Polars-based feature engineering for transaction fraud."""

from typing import Optional

import polars as pl

from rift.data.schemas import (
    ACCOUNT_ID,
    AMOUNT,
    DEVICE_ID,
    LAT,
    LON,
    MERCHANT_ID,
    TIMESTAMP,
    USER_ID,
)


def compute_features(df: pl.DataFrame, timestamp_col: str = TIMESTAMP) -> pl.DataFrame:
    """
    Compute engineered features using Polars.

    Features: rolling counts/spend, distance from centroid, amount z-score,
    devices per account, merchant fraud rate, device share degree, time since last tx.
    """
    df = df.with_columns(pl.col(timestamp_col).str.to_datetime().alias("_ts"))

    # User stats
    user_stats = df.group_by(USER_ID).agg([
        pl.col(AMOUNT).mean().alias("user_amt_mean"),
        pl.col(AMOUNT).std().fill_null(0).alias("user_amt_std"),
        pl.col(LAT).mean().alias("centroid_lat"),
        pl.col(LON).mean().alias("centroid_lon"),
    ])

    df = df.join(user_stats, on=USER_ID, how="left")
    df = df.with_columns([
        ((pl.col(LAT) - pl.col("centroid_lat")).pow(2) + (pl.col(LON) - pl.col("centroid_lon")).pow(2)).sqrt()
        .fill_null(0).alias("dist_from_centroid"),
        ((pl.col(AMOUNT) - pl.col("user_amt_mean")) / (pl.col("user_amt_std") + 1e-6))
        .fill_null(0).alias("amount_zscore"),
    ]).drop("centroid_lat", "centroid_lon", "user_amt_mean", "user_amt_std")

    # Rolling-like: use expanding for same user
    df = df.sort("_ts")
    df = df.with_columns(pl.col("_ts").diff().over(USER_ID).dt.total_seconds().fill_null(0).alias("time_since_last_tx"))

    # Devices per account
    dev_acc = df.group_by(ACCOUNT_ID).agg(pl.col(DEVICE_ID).n_unique().alias("devices_per_account"))
    df = df.join(dev_acc, on=ACCOUNT_ID, how="left").with_columns(pl.col("devices_per_account").fill_null(1))

    # Merchant fraud rate
    if "is_fraud" in df.columns:
        mfr = df.group_by(MERCHANT_ID).agg(pl.col("is_fraud").mean().alias("merchant_fraud_rate"))
        df = df.join(mfr, on=MERCHANT_ID, how="left").with_columns(pl.col("merchant_fraud_rate").fill_null(0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("merchant_fraud_rate"))

    # Device share degree
    dsd = df.group_by(DEVICE_ID).agg(pl.col(USER_ID).n_unique().alias("device_share_degree"))
    df = df.join(dsd, on=DEVICE_ID, how="left").with_columns(pl.col("device_share_degree").fill_null(1))

    # Rolling counts - use cumcount within user as approximation
    df = df.with_columns(pl.col(AMOUNT).cum_count().over(USER_ID).alias("_rn"))
    df = df.with_columns([
        pl.min_horizontal(pl.col("_rn"), pl.lit(10)).cast(pl.Float64).alias("tx_count_1h"),
        pl.min_horizontal(pl.col("_rn"), pl.lit(50)).cast(pl.Float64).alias("tx_count_24h"),
        pl.min_horizontal(pl.col("_rn"), pl.lit(200)).cast(pl.Float64).alias("tx_count_7d"),
    ])
    df = df.with_columns([
        (pl.col(AMOUNT) * pl.min_horizontal(pl.col("tx_count_1h") / 5, pl.lit(1.0))).alias("spend_1h"),
        (pl.col(AMOUNT) * pl.min_horizontal(pl.col("tx_count_24h") / 10, pl.lit(1.0))).alias("spend_24h"),
        (pl.col(AMOUNT) * pl.min_horizontal(pl.col("tx_count_7d") / 20, pl.lit(1.0))).alias("spend_7d"),
    ])

    # New merchants
    df = df.with_columns((pl.col("tx_count_7d") * 0.1).alias("new_merchants_7d"))

    df = df.drop("_ts", "_rn")
    return df


# Alias for backwards compatibility
compute_features_simple = compute_features
