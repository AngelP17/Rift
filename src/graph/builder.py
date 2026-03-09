"""Graph builder: constructs heterogeneous transaction graph from DataFrame."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from utils.logging import get_logger

log = get_logger(__name__)


class GraphData:
    """Lightweight container for heterogeneous graph data.

    Avoids hard dependency on torch_geometric at import time; can be
    converted to HeteroData when needed.
    """

    def __init__(self) -> None:
        self.node_types: dict[str, int] = {}
        self.node_features: dict[str, torch.Tensor] = {}
        self.node_ids: dict[str, list[str]] = {}
        self.edge_index: dict[tuple[str, str, str], torch.Tensor] = {}
        self.edge_attr: dict[tuple[str, str, str], torch.Tensor | None] = {}
        self.labels: torch.Tensor | None = None
        self.tx_indices: np.ndarray | None = None

    def num_nodes(self, ntype: str) -> int:
        return self.node_types.get(ntype, 0)

    def num_edges(self, etype: tuple[str, str, str]) -> int:
        if etype in self.edge_index:
            return self.edge_index[etype].shape[1]
        return 0


def build_graph(df: pl.DataFrame, feature_cols: list[str] | None = None) -> GraphData:
    """Build a heterogeneous graph from transaction DataFrame.

    Node types: user, merchant, device, account, transaction
    Edge types: user->transaction, transaction->merchant, transaction->device,
                transaction->account, user->device, user->merchant, account->device
    """
    log.info("building_graph", n_txns=len(df))
    g = GraphData()

    user_ids = df["user_id"].unique().sort().to_list()
    merchant_ids = df["merchant_id"].unique().sort().to_list()
    device_ids = df["device_id"].unique().sort().to_list()
    account_ids = df["account_id"].unique().sort().to_list()
    tx_ids = df["tx_id"].to_list()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    merchant_map = {mid: i for i, mid in enumerate(merchant_ids)}
    device_map = {did: i for i, did in enumerate(device_ids)}
    account_map = {aid: i for i, aid in enumerate(account_ids)}
    tx_map = {tid: i for i, tid in enumerate(tx_ids)}

    g.node_types = {
        "user": len(user_ids),
        "merchant": len(merchant_ids),
        "device": len(device_ids),
        "account": len(account_ids),
        "transaction": len(tx_ids),
    }
    g.node_ids = {
        "user": user_ids,
        "merchant": merchant_ids,
        "device": device_ids,
        "account": account_ids,
        "transaction": tx_ids,
    }

    if feature_cols:
        available = [c for c in feature_cols if c in df.columns]
        feat_np = df.select(available).to_numpy().astype(np.float32)
        g.node_features["transaction"] = torch.from_numpy(feat_np)
    else:
        g.node_features["transaction"] = torch.ones(len(tx_ids), 1)

    g.node_features["user"] = torch.ones(len(user_ids), 1)
    g.node_features["merchant"] = torch.ones(len(merchant_ids), 1)
    g.node_features["device"] = torch.ones(len(device_ids), 1)
    g.node_features["account"] = torch.ones(len(account_ids), 1)

    if "is_fraud" in df.columns:
        g.labels = torch.tensor(df["is_fraud"].to_list(), dtype=torch.long)

    _build_edges(df, g, user_map, merchant_map, device_map, account_map, tx_map)

    log.info(
        "graph_built",
        nodes={k: v for k, v in g.node_types.items()},
        edges={str(k): g.num_edges(k) for k in g.edge_index},
    )
    return g


def _build_edges(
    df: pl.DataFrame,
    g: GraphData,
    user_map: dict, merchant_map: dict,
    device_map: dict, account_map: dict, tx_map: dict,
) -> None:
    users = df["user_id"].to_list()
    merchants = df["merchant_id"].to_list()
    devices = df["device_id"].to_list()
    accounts = df["account_id"].to_list()
    txns = df["tx_id"].to_list()

    u2t_src, u2t_dst = [], []
    t2m_src, t2m_dst = [], []
    t2d_src, t2d_dst = [], []
    t2a_src, t2a_dst = [], []

    u2d_pairs: set[tuple[int, int]] = set()
    u2m_pairs: set[tuple[int, int]] = set()
    a2d_pairs: set[tuple[int, int]] = set()

    for i in range(len(df)):
        ui = user_map[users[i]]
        mi = merchant_map[merchants[i]]
        di = device_map[devices[i]]
        ai = account_map[accounts[i]]
        ti = tx_map[txns[i]]

        u2t_src.append(ui)
        u2t_dst.append(ti)
        t2m_src.append(ti)
        t2m_dst.append(mi)
        t2d_src.append(ti)
        t2d_dst.append(di)
        t2a_src.append(ti)
        t2a_dst.append(ai)

        u2d_pairs.add((ui, di))
        u2m_pairs.add((ui, mi))
        a2d_pairs.add((ai, di))

    def _to_tensor(src: list[int], dst: list[int]) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)

    g.edge_index[("user", "initiates", "transaction")] = _to_tensor(u2t_src, u2t_dst)
    g.edge_index[("transaction", "at", "merchant")] = _to_tensor(t2m_src, t2m_dst)
    g.edge_index[("transaction", "via", "device")] = _to_tensor(t2d_src, t2d_dst)
    g.edge_index[("transaction", "from", "account")] = _to_tensor(t2a_src, t2a_dst)

    if u2d_pairs:
        src, dst = zip(*u2d_pairs)
        g.edge_index[("user", "uses", "device")] = _to_tensor(list(src), list(dst))
    if u2m_pairs:
        src, dst = zip(*u2m_pairs)
        g.edge_index[("user", "shops_at", "merchant")] = _to_tensor(list(src), list(dst))
    if a2d_pairs:
        src, dst = zip(*a2d_pairs)
        g.edge_index[("account", "linked", "device")] = _to_tensor(list(src), list(dst))
