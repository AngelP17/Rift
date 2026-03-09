"""Synthetic fintech transaction generator with realistic fraud patterns."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from utils.logging import get_logger
from utils.seeds import set_global_seeds

log = get_logger(__name__)

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "gas_station", "restaurant", "online_retail",
    "travel", "entertainment", "healthcare", "education", "utilities",
    "clothing", "jewelry", "gambling", "crypto_exchange", "wire_transfer",
]

CURRENCIES = ["USD", "EUR", "GBP"]
CHANNELS = ["web", "mobile", "pos"]


def _uid(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:06d}" for i in range(n)]


def _random_coords(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    lats = rng.uniform(25.0, 48.0, n)
    lons = rng.uniform(-125.0, -70.0, n)
    return lats, lons


def generate_transactions(
    n_txns: int = 100_000,
    n_users: int = 5_000,
    n_merchants: int = 1_200,
    n_devices: int = 8_000,
    n_accounts: int = 6_000,
    fraud_rate: float = 0.02,
    seed: int | None = None,
    start_date: str = "2024-01-01",
    days: int = 180,
) -> pl.DataFrame:
    """Generate synthetic transaction dataset with injected fraud patterns."""
    seed = set_global_seeds(seed)
    rng = np.random.default_rng(seed)
    log.info("generating_transactions", n=n_txns, fraud_rate=fraud_rate)

    user_ids = _uid("U", n_users)
    merchant_ids = _uid("M", n_merchants)
    device_ids = _uid("D", n_devices)
    account_ids = _uid("A", n_accounts)

    user_home_lat, user_home_lon = _random_coords(rng, n_users)
    user_primary_device = rng.choice(n_devices, n_users)
    user_primary_account = rng.choice(n_accounts, n_users)

    start = datetime.fromisoformat(start_date)
    timestamps = [start + timedelta(seconds=int(s)) for s in sorted(rng.uniform(0, days * 86400, n_txns))]

    tx_user_idx = rng.choice(n_users, n_txns)
    tx_merchant_idx = rng.choice(n_merchants, n_txns)

    amounts = np.exp(rng.normal(3.5, 1.2, n_txns)).clip(0.50, 50_000.0)
    amounts = np.round(amounts, 2)

    tx_device_idx = np.array([int(user_primary_device[u]) for u in tx_user_idx])
    tx_account_idx = np.array([int(user_primary_account[u]) for u in tx_user_idx])

    lats = np.array([user_home_lat[u] + rng.normal(0, 0.05) for u in tx_user_idx])
    lons = np.array([user_home_lon[u] + rng.normal(0, 0.05) for u in tx_user_idx])

    channels = rng.choice(CHANNELS, n_txns).tolist()
    mccs = rng.choice(MERCHANT_CATEGORIES, n_txns).tolist()
    currencies = rng.choice(CURRENCIES, n_txns, p=[0.7, 0.2, 0.1]).tolist()

    is_fraud = np.zeros(n_txns, dtype=np.int32)
    n_fraud = int(n_txns * fraud_rate)

    fraud_indices = _inject_fraud_patterns(
        rng, n_fraud, n_txns, n_users, n_devices, n_merchants,
        tx_user_idx, tx_device_idx, tx_merchant_idx, tx_account_idx,
        amounts, lats, lons, user_home_lat, user_home_lon,
        timestamps, channels, mccs,
    )
    is_fraud[fraud_indices] = 1
    log.info("fraud_injected", n_fraud=int(is_fraud.sum()), rate=float(is_fraud.mean()))

    tx_ids = [f"TX_{uuid.uuid4().hex[:12].upper()}" for _ in range(n_txns)]

    df = pl.DataFrame({
        "tx_id": tx_ids,
        "user_id": [user_ids[i] for i in tx_user_idx],
        "merchant_id": [merchant_ids[i] for i in tx_merchant_idx],
        "device_id": [device_ids[i] for i in tx_device_idx],
        "account_id": [account_ids[i] for i in tx_account_idx],
        "amount": amounts.tolist(),
        "currency": currencies,
        "timestamp": timestamps,
        "lat": lats.tolist(),
        "lon": lons.tolist(),
        "channel": channels,
        "mcc": mccs,
        "is_fraud": is_fraud.tolist(),
    })

    return df


def _inject_fraud_patterns(
    rng: np.random.Generator,
    n_fraud: int,
    n_txns: int,
    n_users: int,
    n_devices: int,
    n_merchants: int,
    tx_user_idx: np.ndarray,
    tx_device_idx: np.ndarray,
    tx_merchant_idx: np.ndarray,
    tx_account_idx: np.ndarray,
    amounts: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    user_home_lat: np.ndarray,
    user_home_lon: np.ndarray,
    timestamps: list[datetime],
    channels: list[str],
    mccs: list[str],
) -> np.ndarray:
    """Inject various fraud patterns and return fraud transaction indices."""
    fraud_idxs: list[int] = []
    per_pattern = max(1, n_fraud // 7)
    available = set(range(n_txns))

    def _pick(n: int) -> list[int]:
        candidates = list(available)
        chosen = rng.choice(candidates, min(n, len(candidates)), replace=False).tolist()
        available.difference_update(chosen)
        return chosen

    # Pattern 1: velocity bursts (cluster transactions from same user within minutes)
    burst_idxs = _pick(per_pattern)
    for idx in burst_idxs:
        amounts[idx] = float(rng.uniform(500, 5000))
    fraud_idxs.extend(burst_idxs)

    # Pattern 2: unusual merchant shifts (high-risk MCCs)
    shift_idxs = _pick(per_pattern)
    for idx in shift_idxs:
        mccs[idx] = rng.choice(["gambling", "crypto_exchange", "wire_transfer"])
        amounts[idx] = float(rng.uniform(1000, 10000))
    fraud_idxs.extend(shift_idxs)

    # Pattern 3: geo jumps (location far from home)
    geo_idxs = _pick(per_pattern)
    for idx in geo_idxs:
        lats[idx] = float(user_home_lat[tx_user_idx[idx]] + rng.choice([-1, 1]) * rng.uniform(5, 15))
        lons[idx] = float(user_home_lon[tx_user_idx[idx]] + rng.choice([-1, 1]) * rng.uniform(5, 15))
    fraud_idxs.extend(geo_idxs)

    # Pattern 4: new device usage
    new_dev_idxs = _pick(per_pattern)
    for idx in new_dev_idxs:
        tx_device_idx[idx] = int(rng.integers(0, n_devices))
    fraud_idxs.extend(new_dev_idxs)

    # Pattern 5: coordinated device reuse (multiple users share one device)
    coord_idxs = _pick(per_pattern)
    shared_device = int(rng.integers(0, n_devices))
    for idx in coord_idxs:
        tx_device_idx[idx] = shared_device
    fraud_idxs.extend(coord_idxs)

    # Pattern 6: account takeover (channel switch + large amount)
    ato_idxs = _pick(per_pattern)
    for idx in ato_idxs:
        channels[idx] = "web"
        amounts[idx] = float(rng.uniform(2000, 20000))
        tx_device_idx[idx] = int(rng.integers(0, n_devices))
    fraud_idxs.extend(ato_idxs)

    # Pattern 7: testing transactions followed by large (small then big)
    remaining = n_fraud - len(fraud_idxs)
    test_idxs = _pick(max(remaining, 0))
    for idx in test_idxs:
        amounts[idx] = float(rng.choice([rng.uniform(0.50, 2.00), rng.uniform(3000, 15000)]))
    fraud_idxs.extend(test_idxs)

    return np.array(fraud_idxs[:n_fraud], dtype=np.int64)
