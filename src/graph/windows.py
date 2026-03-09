"""Windowed graph construction for temporal evaluation."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from graph.builder import GraphData, build_graph
from utils.logging import get_logger

log = get_logger(__name__)


def build_windowed_graph(
    df: pl.DataFrame,
    reference_time: object,
    window_days: int = 7,
    feature_cols: list[str] | None = None,
) -> GraphData:
    """Build a graph using only transactions within a time window."""
    cutoff = reference_time - timedelta(days=window_days)
    windowed = df.filter(
        (pl.col("timestamp") >= cutoff) & (pl.col("timestamp") <= reference_time)
    )
    log.info("windowed_graph", window_days=window_days, n_txns=len(windowed))
    return build_graph(windowed, feature_cols)


def build_rolling_graphs(
    df: pl.DataFrame,
    window_days: int = 7,
    step_days: int = 7,
    feature_cols: list[str] | None = None,
) -> list[tuple[object, GraphData]]:
    """Build a sequence of rolling-window graphs."""
    sorted_df = df.sort("timestamp")
    min_ts = sorted_df["timestamp"].min()
    max_ts = sorted_df["timestamp"].max()

    graphs = []
    current = min_ts + timedelta(days=window_days)

    while current <= max_ts:
        g = build_windowed_graph(sorted_df, current, window_days, feature_cols)
        if g.node_types.get("transaction", 0) > 0:
            graphs.append((current, g))
        current += timedelta(days=step_days)

    log.info("rolling_graphs_built", count=len(graphs))
    return graphs
