"""Main feature engineering orchestrator using Polars."""

from __future__ import annotations

import polars as pl

from features.aggregates import compute_aggregate_features
from features.temporal import compute_temporal_features
from utils.logging import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS: list[str] = [
    "tx_count_1h", "tx_count_24h", "tx_count_7d",
    "spend_1h", "spend_24h", "spend_7d",
    "dist_from_centroid",
    "new_merchants_24h", "new_merchants_7d",
    "devices_per_account",
    "merchant_fraud_rate",
    "device_sharing_degree",
    "time_since_last_tx",
    "amount_zscore",
    "amount", "lat", "lon",
    "channel_web", "channel_mobile", "channel_pos",
]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build all engineered features from raw transaction data.

    Returns the original DataFrame augmented with feature columns.
    """
    log.info("building_features", n_rows=len(df))
    df = df.sort("timestamp")

    df = compute_temporal_features(df)
    df = compute_aggregate_features(df)

    df = _encode_channel(df)
    df = _fill_nulls(df)

    log.info("features_built", n_features=len(FEATURE_COLUMNS))
    return df


def _encode_channel(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("channel") == "web").cast(pl.Int32).alias("channel_web"),
        (pl.col("channel") == "mobile").cast(pl.Int32).alias("channel_mobile"),
        (pl.col("channel") == "pos").cast(pl.Int32).alias("channel_pos"),
    )


def _fill_nulls(df: pl.DataFrame) -> pl.DataFrame:
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))
    return df


def get_feature_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Extract only the feature columns needed for model training."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df.select(available)
