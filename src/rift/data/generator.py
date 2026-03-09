from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from rift.utils.seeds import seed_everything


CHANNELS = ["web", "mobile", "pos"]
MCCS = ["grocery", "travel", "electronics", "gaming", "general", "luxury"]
CURRENCIES = ["USD"]


def _pick_ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{idx:05d}" for idx in range(count)]


def generate_transactions(
    txns: int = 10_000,
    users: int = 1_000,
    merchants: int = 200,
    fraud_rate: float = 0.02,
    seed: int = 7,
) -> pl.DataFrame:
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    user_ids = _pick_ids("user", users)
    merchant_ids = _pick_ids("merchant", merchants)
    device_ids = _pick_ids("device", max(users // 2, 50))
    account_ids = _pick_ids("acct", max(users, 100))

    user_home = {
        user_id: (
            float(rng.uniform(25.0, 48.0)),
            float(rng.uniform(-123.0, -71.0)),
        )
        for user_id in user_ids
    }
    user_device = {user_id: rng.choice(device_ids) for user_id in user_ids}
    user_account = {user_id: rng.choice(account_ids) for user_id in user_ids}

    start = datetime(2025, 1, 1, 0, 0, 0)
    current_times = {
        user_id: start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 5)))
        for user_id in user_ids
    }

    rows: list[dict[str, object]] = []
    for idx in range(txns):
        user_id = rng.choice(user_ids)
        merchant_id = rng.choice(merchant_ids)
        base_device = user_device[user_id]
        base_account = user_account[user_id]
        base_lat, base_lon = user_home[user_id]

        current_times[user_id] = current_times[user_id] + timedelta(
            minutes=int(rng.integers(5, 300))
        )
        timestamp = current_times[user_id]

        amount = max(1.0, float(rng.lognormal(mean=3.4, sigma=0.8)))
        lat = float(base_lat + rng.normal(0, 0.3))
        lon = float(base_lon + rng.normal(0, 0.3))
        device_id = str(base_device)
        account_id = str(base_account)
        channel = str(rng.choice(CHANNELS))
        mcc = str(rng.choice(MCCS))
        is_fraud = 0

        if rng.random() < fraud_rate:
            is_fraud = 1
            fraud_pattern = int(rng.integers(0, 5))
            if fraud_pattern == 0:
                amount *= float(rng.uniform(4.0, 8.0))
            elif fraud_pattern == 1:
                lat = float(rng.uniform(0.0, 55.0))
                lon = float(rng.uniform(-130.0, -10.0))
            elif fraud_pattern == 2:
                device_id = str(rng.choice(device_ids))
                merchant_id = str(rng.choice(merchant_ids))
            elif fraud_pattern == 3:
                amount = float(rng.uniform(1.0, 5.0))
                current_times[user_id] = current_times[user_id] + timedelta(minutes=2)
            else:
                account_id = str(rng.choice(account_ids))
                device_id = str(rng.choice(device_ids))
                channel = "web"
                amount *= float(rng.uniform(2.5, 5.0))

        rows.append(
            {
                "tx_id": f"tx_{idx:07d}",
                "user_id": str(user_id),
                "merchant_id": str(merchant_id),
                "device_id": device_id,
                "account_id": account_id,
                "amount": round(amount, 2),
                "currency": CURRENCIES[0],
                "timestamp": timestamp,
                "lat": lat,
                "lon": lon,
                "channel": channel,
                "mcc": mcc,
                "is_fraud": is_fraud,
            }
        )

    return pl.DataFrame(rows).sort("timestamp")
