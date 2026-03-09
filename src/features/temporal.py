"""Temporal features: rolling windows, geo distance, time gaps."""

from __future__ import annotations

from datetime import timedelta
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import polars as pl

from utils.logging import get_logger

log = get_logger(__name__)


def compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all temporal features. DataFrame must be sorted by timestamp."""
    df = _rolling_counts_and_spend(df)
    df = _time_since_last(df)
    df = _distance_from_centroid(df)
    return df


def _rolling_counts_and_spend(df: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling transaction count and spend over 1h, 24h, 7d per user."""
    ts_col = df["timestamp"].to_list()
    user_col = df["user_id"].to_list()
    amount_col = df["amount"].to_list()

    user_history: dict[str, list[tuple[object, float]]] = {}
    results = {
        "tx_count_1h": [], "tx_count_24h": [], "tx_count_7d": [],
        "spend_1h": [], "spend_24h": [], "spend_7d": [],
    }

    windows = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7)}

    for i in range(len(df)):
        uid = user_col[i]
        ts = ts_col[i]
        amt = amount_col[i]

        if uid not in user_history:
            user_history[uid] = []

        hist = user_history[uid]
        for label, delta in windows.items():
            cutoff = ts - delta
            matching = [(t, a) for t, a in hist if t >= cutoff]
            results[f"tx_count_{label}"].append(len(matching))
            results[f"spend_{label}"].append(sum(a for _, a in matching))

        hist.append((ts, amt))

    for col_name, values in results.items():
        df = df.with_columns(pl.Series(col_name, values, dtype=pl.Float64))

    return df


def _time_since_last(df: pl.DataFrame) -> pl.DataFrame:
    """Time in seconds since the user's previous transaction."""
    ts_col = df["timestamp"].to_list()
    user_col = df["user_id"].to_list()

    last_ts: dict[str, object] = {}
    gaps = []

    for i in range(len(df)):
        uid = user_col[i]
        ts = ts_col[i]
        if uid in last_ts:
            delta = (ts - last_ts[uid]).total_seconds()
            gaps.append(delta)
        else:
            gaps.append(0.0)
        last_ts[uid] = ts

    return df.with_columns(pl.Series("time_since_last_tx", gaps, dtype=pl.Float64))


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _distance_from_centroid(df: pl.DataFrame) -> pl.DataFrame:
    """Distance in km from the user's running average location."""
    lat_col = df["lat"].to_list()
    lon_col = df["lon"].to_list()
    user_col = df["user_id"].to_list()

    centroids: dict[str, tuple[list[float], list[float]]] = {}
    dists = []

    for i in range(len(df)):
        uid = user_col[i]
        lat, lon = lat_col[i], lon_col[i]

        if uid not in centroids:
            centroids[uid] = ([], [])

        lat_hist, lon_hist = centroids[uid]
        if lat_hist:
            c_lat = np.mean(lat_hist)
            c_lon = np.mean(lon_hist)
            dists.append(_haversine(lat, lon, c_lat, c_lon))
        else:
            dists.append(0.0)

        lat_hist.append(lat)
        lon_hist.append(lon)

    return df.with_columns(pl.Series("dist_from_centroid", dists, dtype=pl.Float64))
