"""Shared test fixtures for Rift."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generator import generate_transactions
from utils.seeds import set_global_seeds


@pytest.fixture(scope="session")
def small_dataset():
    """Small synthetic dataset for fast testing."""
    set_global_seeds(42)
    return generate_transactions(n_txns=500, n_users=50, n_merchants=20, fraud_rate=0.05, seed=42)


@pytest.fixture(scope="session")
def featured_dataset(small_dataset):
    """Small dataset with features computed."""
    from features.engine import build_features
    return build_features(small_dataset)


@pytest.fixture
def sample_transaction():
    """Single sample transaction dict."""
    return {
        "tx_id": "TX_TEST001",
        "user_id": "U_000001",
        "merchant_id": "M_000001",
        "device_id": "D_000001",
        "account_id": "A_000001",
        "amount": 150.00,
        "currency": "USD",
        "timestamp": "2024-03-15T14:30:00",
        "lat": 37.7749,
        "lon": -122.4194,
        "channel": "web",
        "mcc": "online_retail",
        "is_fraud": 0,
    }


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary DuckDB path for testing."""
    return tmp_path / "test_audit.duckdb"
