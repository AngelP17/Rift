from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl


@dataclass
class TransactionGraph:
    tx_ids: list[str]
    edge_index: np.ndarray
    edge_type: list[str]


def build_transaction_graph(
    frame: pl.DataFrame,
    window_days: int | None = 30,
    max_links_per_group: int = 5,
) -> TransactionGraph:
    ordered = frame.sort("timestamp")
    rows = ordered.to_dicts()
    tx_ids = [str(row["tx_id"]) for row in rows]
    tx_index = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    groups: dict[str, defaultdict[str, list[tuple[object, str]]]] = {
        "user": defaultdict(list),
        "merchant": defaultdict(list),
        "device": defaultdict(list),
        "account": defaultdict(list),
    }
    edges: list[tuple[int, int]] = []
    edge_types: list[str] = []
    delta = timedelta(days=window_days) if window_days is not None else None

    for row in rows:
        tx_id = str(row["tx_id"])
        ts = row["timestamp"]
        current_idx = tx_index[tx_id]
        keys = {
            "user": str(row["user_id"]),
            "merchant": str(row["merchant_id"]),
            "device": str(row["device_id"]),
            "account": str(row["account_id"]),
        }
        for edge_type, key in keys.items():
            prior = groups[edge_type][key]
            candidates = prior[-max_links_per_group:]
            for prev_ts, prev_tx_id in candidates:
                if delta is not None and ts - prev_ts > delta:
                    continue
                prev_idx = tx_index[prev_tx_id]
                edges.append((prev_idx, current_idx))
                edges.append((current_idx, prev_idx))
                edge_types.extend([edge_type, edge_type])
            prior.append((ts, tx_id))

    edge_index = np.array(edges, dtype=int).T if edges else np.zeros((2, 0), dtype=int)
    return TransactionGraph(tx_ids=tx_ids, edge_index=edge_index, edge_type=edge_types)
