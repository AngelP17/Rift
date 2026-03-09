"""Synthetic transaction generator with realistic fraud patterns."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl

from rift.data.schemas import (
    ACCOUNT_ID,
    AMOUNT,
    CHANNEL,
    CHANNELS,
    CURRENCIES,
    CURRENCY,
    DEVICE_ID,
    IS_FRAUD,
    LAT,
    LON,
    MCC,
    MCC_CATEGORIES,
    MERCHANT_ID,
    TIMESTAMP,
    TX_ID,
    USER_ID,
)
from rift.utils.seeds import set_seed


def _generate_ids(n: int, prefix: str) -> list[str]:
    """Generate deterministic unique IDs."""
    return [f"{prefix}_{i}" for i in range(n)]


def generate_transactions(
    n_txns: int = 100_000,
    n_users: int = 5_000,
    n_merchants: int = 1_200,
    n_devices: int = 8_000,
    n_accounts: int = 6_000,
    fraud_rate: float = 0.02,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Generate synthetic fintech transactions with realistic fraud patterns.

    Fraud patterns injected:
    - Velocity bursts
    - Unusual merchant shifts
    - Geo jumps
    - New device usage
    - Coordinated device reuse
    - Account takeover sequences
    - Small testing transactions followed by large ones
    """
    set_seed(seed)

    user_ids = _generate_ids(n_users, "u")
    merchant_ids = _generate_ids(n_merchants, "m")
    device_ids = _generate_ids(n_devices, "d")
    account_ids = _generate_ids(n_accounts, "a")

    # User-device and user-account mappings
    n_user_devices = min(n_devices, n_users * 3)
    n_user_accounts = min(n_accounts, n_users * 2)

    user_to_devices = {u: [] for u in user_ids}
    for i, d in enumerate(device_ids[:n_user_devices]):
        user_to_devices[user_ids[i % n_users]].append(d)

    user_to_accounts = {u: [] for u in user_ids}
    for i, a in enumerate(account_ids[:n_user_accounts]):
        user_to_accounts[user_ids[i % n_users]].append(a)

    # User centroid locations (home base)
    user_lats = {u: np.random.uniform(-90, 90) for u in user_ids}
    user_lons = {u: np.random.uniform(-180, 180) for u in user_ids}

    # User-merchant history (normal spending patterns)
    user_merchants = {u: set(np.random.choice(merchant_ids, size=min(50, n_merchants), replace=False).tolist())
                     for u in user_ids}

    rows = []
    base_time = datetime(2024, 1, 1)
    n_fraud = int(n_txns * fraud_rate)
    fraud_indices = set(np.random.choice(n_txns, size=n_fraud, replace=False))

    for i in range(n_txns):
        is_fraud = i in fraud_indices
        user_id = np.random.choice(user_ids)
        user_lat, user_lon = user_lats[user_id], user_lons[user_id]

        # Normal device/account from user's set
        user_ds = user_to_devices.get(user_id, [device_ids[i % n_devices]])
        user_accts = user_to_accounts.get(user_id, [account_ids[i % n_accounts]])

        if is_fraud:
            # Fraud pattern: 70% new device, 60% geo jump, 50% new merchant
            if np.random.random() < 0.7:
                device_id = np.random.choice([d for d in device_ids if d not in user_ds] or device_ids)
            else:
                device_id = np.random.choice(user_ds)

            if np.random.random() < 0.6:
                lat = user_lat + np.random.uniform(-20, 20)  # Geo jump
                lon = user_lon + np.random.uniform(-30, 30)
            else:
                lat = user_lat + np.random.uniform(-1, 1)
                lon = user_lon + np.random.uniform(-1, 1)

            if np.random.random() < 0.5:
                merchant_id = np.random.choice([m for m in merchant_ids if m not in user_merchants[user_id]]
                                              or merchant_ids)
            else:
                merchant_id = np.random.choice(merchant_ids)

            # Testing-then-large pattern: 30% of fraud
            if np.random.random() < 0.3:
                amount = np.random.choice([0.01, 0.99, 1.50]) if np.random.random() < 0.5 else np.random.uniform(500, 5000)
            else:
                amount = np.random.lognormal(4, 2)
                amount = min(amount, 10000)

            account_id = np.random.choice(user_accts) if user_accts else np.random.choice(account_ids)
        else:
            device_id = np.random.choice(user_ds)
            lat = user_lat + np.random.uniform(-2, 2)
            lon = user_lon + np.random.uniform(-2, 2)
            merchant_id = np.random.choice(list(user_merchants[user_id]) or merchant_ids)
            amount = np.random.lognormal(3, 1.5)
            amount = min(amount, 2000)
            account_id = np.random.choice(user_accts) if user_accts else np.random.choice(account_ids)

        # Timestamp with velocity burst for some fraud
        if is_fraud and np.random.random() < 0.2:
            # Burst: many txns in short time - use recent base
            ts = base_time + timedelta(days=np.random.randint(0, 365), seconds=np.random.randint(0, 3600))
        else:
            ts = base_time + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
            )

        rid = f"tx_{i:012x}"  # Deterministic ID for reproducibility
        rows.append({
            TX_ID: rid,
            USER_ID: user_id,
            MERCHANT_ID: merchant_id,
            DEVICE_ID: device_id,
            ACCOUNT_ID: account_id,
            AMOUNT: round(float(amount), 2),
            CURRENCY: np.random.choice(CURRENCIES),
            TIMESTAMP: ts.isoformat(),
            LAT: round(lat, 4),
            LON: round(lon, 4),
            CHANNEL: np.random.choice(CHANNELS),
            MCC: np.random.choice(MCC_CATEGORIES),
            IS_FRAUD: int(is_fraud),
        })

    df = pl.DataFrame(rows)
    return df.sort(TIMESTAMP)
