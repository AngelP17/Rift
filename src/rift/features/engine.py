from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import polars as pl


WINDOWS = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * atan2(sqrt(a), sqrt(1.0 - a))


def build_features(
    frame: pl.DataFrame,
    categorical_mappings: dict[str, dict[str, int]] | None = None,
) -> pl.DataFrame:
    ordered = frame.sort("timestamp")
    rows = ordered.to_dicts()

    user_windows = {name: defaultdict(deque) for name in WINDOWS}
    user_amounts = defaultdict(list)
    user_locations = defaultdict(list)
    user_last_seen: dict[str, datetime] = {}
    account_devices = defaultdict(set)
    user_merchants = defaultdict(lambda: {name: deque() for name in WINDOWS})
    merchant_stats = defaultdict(lambda: {"fraud": 0, "count": 0})
    device_users = defaultdict(set)

    feature_rows: list[dict[str, float | int | str | datetime]] = []
    for row in rows:
        ts: datetime = row["timestamp"]
        user_id = str(row["user_id"])
        account_id = str(row["account_id"])
        merchant_id = str(row["merchant_id"])
        device_id = str(row["device_id"])
        amount = float(row["amount"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        features: dict[str, float | int | str | datetime] = {**row}
        for window_name, delta in WINDOWS.items():
            dq = user_windows[window_name][user_id]
            while dq and ts - dq[0][0] > delta:
                dq.popleft()
            features[f"user_txn_count_{window_name}"] = float(len(dq))
            features[f"user_spend_{window_name}"] = float(sum(item[1] for item in dq))
            dq.append((ts, amount))

            merchant_dq = user_merchants[user_id][window_name]
            while merchant_dq and ts - merchant_dq[0][0] > delta:
                merchant_dq.popleft()
            features[f"new_merchants_{window_name}"] = float(
                len({item[1] for item in merchant_dq if item[1] != merchant_id})
            )
            merchant_dq.append((ts, merchant_id))

        if user_locations[user_id]:
            mean_lat = float(np.mean([coord[0] for coord in user_locations[user_id]]))
            mean_lon = float(np.mean([coord[1] for coord in user_locations[user_id]]))
            features["distance_from_user_centroid_km"] = _haversine(mean_lat, mean_lon, lat, lon)
        else:
            features["distance_from_user_centroid_km"] = 0.0

        if user_last_seen.get(user_id):
            features["seconds_since_last_txn"] = float((ts - user_last_seen[user_id]).total_seconds())
        else:
            features["seconds_since_last_txn"] = 0.0

        if user_amounts[user_id]:
            history = np.array(user_amounts[user_id], dtype=float)
            std = float(history.std()) or 1.0
            features["user_amount_zscore"] = (amount - float(history.mean())) / std
        else:
            features["user_amount_zscore"] = 0.0

        features["devices_per_account"] = float(len(account_devices[account_id]))
        features["device_sharing_degree"] = float(len(device_users[device_id]))
        merchant_count = merchant_stats[merchant_id]["count"]
        merchant_fraud = merchant_stats[merchant_id]["fraud"]
        features["merchant_fraud_prevalence"] = float(merchant_fraud / merchant_count) if merchant_count else 0.0
        features["is_new_device_for_user"] = float(device_id not in {d for d in device_users if user_id in device_users[d]})

        feature_rows.append(features)

        user_amounts[user_id].append(amount)
        user_locations[user_id].append((lat, lon))
        user_last_seen[user_id] = ts
        account_devices[account_id].add(device_id)
        device_users[device_id].add(user_id)
        merchant_stats[merchant_id]["count"] += 1
        merchant_stats[merchant_id]["fraud"] += int(row["is_fraud"])

    feature_frame = pl.DataFrame(feature_rows)
    if categorical_mappings is not None:
        return feature_frame.with_columns(
            pl.col("channel").replace_strict(categorical_mappings["channel"], default=0, return_dtype=pl.UInt32).alias("channel_code"),
            pl.col("mcc").replace_strict(categorical_mappings["mcc"], default=0, return_dtype=pl.UInt32).alias("mcc_code"),
            pl.col("currency").replace_strict(categorical_mappings["currency"], default=0, return_dtype=pl.UInt32).alias("currency_code"),
        )
    return feature_frame.with_columns(
        pl.col("channel").cast(pl.Categorical).to_physical().alias("channel_code"),
        pl.col("mcc").cast(pl.Categorical).to_physical().alias("mcc_code"),
        pl.col("currency").cast(pl.Categorical).to_physical().alias("currency_code"),
    )


def extract_categorical_mappings(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for str_col, code_col in [("channel", "channel_code"), ("mcc", "mcc_code"), ("currency", "currency_code")]:
        pairs = frame.select(str_col, code_col).unique()
        mappings[str_col] = dict(zip(pairs[str_col].to_list(), pairs[code_col].cast(int).to_list()))
    return mappings


def feature_columns(frame: pl.DataFrame) -> list[str]:
    excluded = {
        "tx_id",
        "user_id",
        "merchant_id",
        "device_id",
        "account_id",
        "currency",
        "timestamp",
        "channel",
        "mcc",
        "is_fraud",
    }
    return [column for column in frame.columns if column not in excluded]
