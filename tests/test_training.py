"""Tests for training pipeline."""

import pytest

from rift.data.generator import generate_transactions
from rift.data.splits import SplitStrategy
from rift.models.train import train


def test_train_xgb_tabular():
    df = generate_transactions(n_txns=2000, n_users=200, n_merchants=80, seed=42)
    result = train(df, model_type="xgb_tabular", split=SplitStrategy.RANDOM, seed=42)
    assert "metrics" in result
    assert "pr_auc" in result["metrics"]
    assert result["metrics"]["pr_auc"] >= 0


def test_train_graphsage_xgb():
    df = generate_transactions(n_txns=1500, n_users=100, n_merchants=50, seed=42)
    result = train(df, model_type="graphsage_xgb", split=SplitStrategy.RANDOM, seed=42)
    assert "metrics" in result
    assert "model" in result
