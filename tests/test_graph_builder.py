"""Tests for graph builder."""

import polars as pl
import pytest

from rift.data.generator import generate_transactions
from rift.features.engine import compute_features
from rift.graph.builder import build_homogeneous_transaction_graph


def test_build_homogeneous_graph():
    df = generate_transactions(n_txns=200, n_users=20, n_merchants=15, seed=42)
    df_feat = compute_features(df)
    graph = build_homogeneous_transaction_graph(df_feat, ["dist_from_centroid", "amount_zscore", "devices_per_account"])
    assert graph.x.shape[0] == len(df)
    assert graph.edge_index.shape[0] == 2
    assert hasattr(graph, "tx_map")
