"""Tests for synthetic data generator."""

import pytest

from rift.data.generator import generate_transactions


def test_generate_basic():
    df = generate_transactions(n_txns=1000, n_users=100, n_merchants=50, fraud_rate=0.02, seed=42)
    assert len(df) == 1000
    assert df["is_fraud"].sum() > 0
    assert "tx_id" in df.columns
    assert "user_id" in df.columns
    assert "amount" in df.columns


def test_generate_deterministic():
    df1 = generate_transactions(n_txns=500, seed=42)
    df2 = generate_transactions(n_txns=500, seed=42)
    assert df1["tx_id"].to_list() == df2["tx_id"].to_list()
