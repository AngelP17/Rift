"""Heterogeneous transaction graph builder."""

from typing import Optional

import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData

from rift.data.schemas import (
    ACCOUNT_ID,
    DEVICE_ID,
    MERCHANT_ID,
    TX_ID,
    USER_ID,
)


def build_hetero_graph(
    df: pl.DataFrame,
    feature_cols: list[str],
    window_days: Optional[int] = None,
) -> HeteroData:
    """
    Build heterogeneous graph from transactions.

    Node types: user, merchant, device, transaction, account
    Edge types: user->tx, tx->merchant, tx->device, tx->account, user->device, user->merchant, account->device
    """
    if window_days:
        df = _filter_by_window(df, window_days)

    # Create ID mappings
    users = df[USER_ID].unique().sort()
    merchants = df[MERCHANT_ID].unique().sort()
    devices = df[DEVICE_ID].unique().sort()
    accounts = df[ACCOUNT_ID].unique().sort()
    tx_ids = df[TX_ID].unique().sort()

    user_map = {u: i for i, u in enumerate(users.to_list())}
    merchant_map = {m: i for i, m in enumerate(merchants.to_list())}
    device_map = {d: i for i, d in enumerate(devices.to_list())}
    account_map = {a: i for i, a in enumerate(accounts.to_list())}
    tx_map = {t: i for i, t in enumerate(tx_ids.to_list())}

    n_users = len(user_map)
    n_merchants = len(merchant_map)
    n_devices = len(device_map)
    n_accounts = len(account_map)
    n_tx = len(tx_map)

    # user -> transaction
    ut_src, ut_dst = [], []
    for row in df.iter_rows(named=True):
        uid = user_map.get(row[USER_ID])
        tid = tx_map.get(row[TX_ID])
        if uid is not None and tid is not None:
            ut_src.append(uid)
            ut_dst.append(tid)
    edge_user_tx = (torch.tensor(ut_src), torch.tensor(ut_dst))

    # transaction -> merchant
    tm_src, tm_dst = [], []
    for row in df.iter_rows(named=True):
        tid = tx_map.get(row[TX_ID])
        mid = merchant_map.get(row[MERCHANT_ID])
        if tid is not None and mid is not None:
            tm_src.append(tid)
            tm_dst.append(mid)
    edge_tx_merchant = (torch.tensor(tm_src), torch.tensor(tm_dst))

    # transaction -> device
    td_src, td_dst = [], []
    for row in df.iter_rows(named=True):
        tid = tx_map.get(row[TX_ID])
        did = device_map.get(row[DEVICE_ID])
        if tid is not None and did is not None:
            td_src.append(tid)
            td_dst.append(did)
    edge_tx_device = (torch.tensor(td_src), torch.tensor(td_dst))

    # transaction -> account
    ta_src, ta_dst = [], []
    for row in df.iter_rows(named=True):
        tid = tx_map.get(row[TX_ID])
        aid = account_map.get(row[ACCOUNT_ID])
        if tid is not None and aid is not None:
            ta_src.append(tid)
            ta_dst.append(aid)
    edge_tx_account = (torch.tensor(ta_src), torch.tensor(ta_dst))

    # user -> device (from tx data)
    ud_pairs = df.select([USER_ID, DEVICE_ID]).unique()
    ud_src, ud_dst = [], []
    for row in ud_pairs.iter_rows(named=True):
        uid = user_map.get(row[USER_ID])
        did = device_map.get(row[DEVICE_ID])
        if uid is not None and did is not None:
            ud_src.append(uid)
            ud_dst.append(did)
    edge_user_device = (torch.tensor(ud_src), torch.tensor(ud_dst))

    # user -> merchant
    um_pairs = df.select([USER_ID, MERCHANT_ID]).unique()
    um_src, um_dst = [], []
    for row in um_pairs.iter_rows(named=True):
        uid = user_map.get(row[USER_ID])
        mid = merchant_map.get(row[MERCHANT_ID])
        if uid is not None and mid is not None:
            um_src.append(uid)
            um_dst.append(mid)
    edge_user_merchant = (torch.tensor(um_src), torch.tensor(um_dst))

    # account -> device
    ad_pairs = df.select([ACCOUNT_ID, DEVICE_ID]).unique()
    ad_src, ad_dst = [], []
    for row in ad_pairs.iter_rows(named=True):
        aid = account_map.get(row[ACCOUNT_ID])
        did = device_map.get(row[DEVICE_ID])
        if aid is not None and did is not None:
            ad_src.append(aid)
            ad_dst.append(did)
    edge_account_device = (torch.tensor(ad_src), torch.tensor(ad_dst))

    # Node features - transaction nodes get engineered features
    avail_cols = [c for c in feature_cols if c in df.columns]
    if avail_cols:
        tx_feat = df.select(avail_cols).to_numpy()
        tx_feat = np.nan_to_num(tx_feat, nan=0.0)
        tx_feat = torch.tensor(tx_feat, dtype=torch.float32)
    else:
        # Fallback: one-hot or random
        tx_feat = torch.randn(n_tx, 8)

    # Other nodes: simple one-hot or learned later
    user_feat = torch.ones(n_users, tx_feat.shape[1]) * 0.5  # Placeholder
    merchant_feat = torch.ones(n_merchants, tx_feat.shape[1]) * 0.5
    device_feat = torch.ones(n_devices, tx_feat.shape[1]) * 0.5
    account_feat = torch.ones(n_accounts, tx_feat.shape[1]) * 0.5

    # Ensure same feature dim
    data = HeteroData()
    data["transaction"].x = tx_feat
    data["user"].x = user_feat[:, : tx_feat.shape[1]] if user_feat.shape[1] >= tx_feat.shape[1] else torch.nn.functional.pad(
        user_feat, (0, tx_feat.shape[1] - user_feat.shape[1])
    )
    data["merchant"].x = merchant_feat[:, : tx_feat.shape[1]] if merchant_feat.shape[1] >= tx_feat.shape[1] else torch.nn.functional.pad(
        merchant_feat, (0, tx_feat.shape[1] - merchant_feat.shape[1])
    )
    data["device"].x = device_feat[:, : tx_feat.shape[1]]
    data["account"].x = account_feat[:, : tx_feat.shape[1]]

    data["user", "to", "transaction"].edge_index = edge_user_tx
    data["transaction", "to", "merchant"].edge_index = edge_tx_merchant
    data["transaction", "to", "device"].edge_index = edge_tx_device
    data["transaction", "to", "account"].edge_index = edge_tx_account
    data["user", "to", "device"].edge_index = edge_user_device
    data["user", "to", "merchant"].edge_index = edge_user_merchant
    data["account", "to", "device"].edge_index = edge_account_device

    data["transaction", "rev", "user"].edge_index = torch.stack([edge_user_tx[1], edge_user_tx[0]])
    data["merchant", "rev", "transaction"].edge_index = torch.stack([edge_tx_merchant[1], edge_tx_merchant[0]])
    data["device", "rev", "transaction"].edge_index = torch.stack([edge_tx_device[1], edge_tx_device[0]])
    data["account", "rev", "transaction"].edge_index = torch.stack([edge_tx_account[1], edge_tx_account[0]])

    data.tx_map = tx_map
    data.user_map = user_map
    data.merchant_map = merchant_map
    data.device_map = device_map
    data.account_map = account_map

    return data


def build_homogeneous_transaction_graph(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> "torch_geometric.data.Data":
    """Build simplified homogeneous graph (transaction nodes only, connected via user/merchant)."""
    from torch_geometric.data import Data

    tx_ids = df[TX_ID].unique().sort().to_list()
    tx_map = {t: i for i, t in enumerate(tx_ids)}
    n_tx = len(tx_map)

    # Connect transactions that share user or merchant
    user_to_tx = df.group_by(USER_ID).agg(pl.col(TX_ID).alias("tx_list"))
    edges_src, edges_dst = [], []
    for row in user_to_tx.iter_rows(named=True):
        txs = row["tx_list"]
        if isinstance(txs, (list, tuple)):
            ids = [tx_map[t] for t in txs if t in tx_map]
        else:
            ids = [tx_map[txs]] if txs in tx_map else []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges_src.extend([ids[i], ids[j]])
                edges_dst.extend([ids[j], ids[i]])

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    avail_cols = [c for c in feature_cols if c in df.columns]
    if avail_cols:
        x = torch.tensor(np.nan_to_num(df.select(avail_cols).to_numpy(), nan=0.0), dtype=torch.float32)
    else:
        x = torch.randn(n_tx, 8)

    return Data(x=x, edge_index=edge_index, tx_map=tx_map)


def _filter_by_window(df: pl.DataFrame, window_days: int) -> pl.DataFrame:
    """Filter transactions to rolling window."""
    df = df.with_columns(pl.col("timestamp").str.to_datetime())
    max_ts = df["timestamp"].max()
    from datetime import timedelta
    cutoff = max_ts - timedelta(days=window_days)
    return df.filter(pl.col("timestamp") >= cutoff)
