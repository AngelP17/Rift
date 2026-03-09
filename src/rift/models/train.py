"""Training pipeline for Rift models."""

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import polars as pl
import torch

from rift.data.schemas import FEATURE_COLUMNS, IS_FRAUD
from rift.data.splits import SplitStrategy, train_val_test_split
from rift.features.engine import compute_features
from rift.graph.builder import build_homogeneous_transaction_graph
from rift.models.baseline_xgb import TabularXGBoost
from rift.models.calibrate import CalibrationMethod, Calibrator
from rift.models.conformal import conformal_triage
from rift.models.ensemble import GraphXGBoostEnsemble
from rift.models.metrics import brier, ece, pr_auc, recall_at_fpr
from rift.utils.io import save_json
from rift.utils.seeds import set_seed


def train(
    df: pl.DataFrame,
    model_type: Literal["xgb_tabular", "graphsage_only", "graphsage_xgb", "gat_xgb"] = "graphsage_xgb",
    split: SplitStrategy = SplitStrategy.CHRONOLOGICAL,
    time_window_days: Optional[int] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Train model and return metrics + artifact paths."""
    set_seed(seed)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feat_cols:
        feat_cols = FEATURE_COLUMNS

    df_feat = compute_features(df)
    train_df, val_df, test_df = train_val_test_split(
        df_feat, strategy=split, time_col="timestamp", seed=seed
    )

    X_train = train_df.select(feat_cols).to_numpy()
    y_train = train_df[IS_FRAUD].to_numpy()
    X_val = val_df.select(feat_cols).to_numpy()
    y_val = val_df[IS_FRAUD].to_numpy()
    X_test = test_df.select(feat_cols).to_numpy()
    y_test = test_df[IS_FRAUD].to_numpy()

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    result = {"model_type": model_type, "split": split.value, "metrics": {}}

    if model_type == "xgb_tabular":
        model = TabularXGBoost()
        model.fit(X_train, y_train, feature_names=feat_cols)
        raw_val = model.predict_proba(X_val)
        raw_test = model.predict_proba(X_test)

    elif model_type in ("graphsage_xgb", "gat_xgb"):
        graph_train = build_homogeneous_transaction_graph(train_df, feat_cols)
        x = graph_train.x
        edge_index = graph_train.edge_index
        encoder = "graphsage" if "graphsage" in model_type else "gat"
        ensemble = GraphXGBoostEnsemble(encoder_type=encoder)
        # Graph nodes are in sorted tx_id order; align labels
        inv_tx_map = {v: k for k, v in graph_train.tx_map.items()}
        label_by_tx = dict(zip(train_df["tx_id"].to_list(), train_df[IS_FRAUD].to_numpy()))
        y_node = np.array([label_by_tx.get(inv_tx_map.get(i), 0) for i in range(x.shape[0])])
        ensemble.fit(x, edge_index, X_train[: x.shape[0]], y_node, feature_names=feat_cols)

        graph_val = build_homogeneous_transaction_graph(val_df, feat_cols)
        raw_val = ensemble.predict_proba(graph_val.x, graph_val.edge_index, X_val)
        graph_test = build_homogeneous_transaction_graph(test_df, feat_cols)
        raw_test = ensemble.predict_proba(graph_test.x, graph_test.edge_index, X_test)
        model = ensemble

    else:
        # graphsage_only - GNN + simple MLP head
        from rift.models.graphsage import FraudGraphSAGE
        graph_train = build_homogeneous_transaction_graph(train_df, feat_cols)
        gnn = FraudGraphSAGE(graph_train.x.shape[1], 32, 16)
        opt = torch.optim.Adam(gnn.parameters(), lr=0.01)
        x = graph_train.x
        edge_index = graph_train.edge_index
        inv_tx_map = {v: k for k, v in graph_train.tx_map.items()}
        label_by_tx = dict(zip(train_df["tx_id"].to_list(), train_df[IS_FRAUD].to_numpy()))
        y_node = np.array([label_by_tx.get(inv_tx_map.get(i), 0) for i in range(x.shape[0])])
        if len(y_node) < x.shape[0]:
            y_node = np.pad(y_node, (0, x.shape[0] - len(y_node)))
        y_t = torch.tensor(y_node, dtype=torch.float32).unsqueeze(1)
        for _ in range(100):
            opt.zero_grad()
            out = gnn(x, edge_index)
            pred = torch.sigmoid(out.mean(dim=1, keepdim=True))
            loss = torch.nn.functional.binary_cross_entropy(pred, y_t)
            loss.backward()
            opt.step()
        gnn.eval()
        with torch.no_grad():
            emb = gnn(graph_train.x, graph_train.edge_index)
        # Train small XGB on embeddings only
        model = TabularXGBoost()
        model.fit(emb.numpy(), y_train[: emb.shape[0]], feature_names=feat_cols)
        graph_val = build_homogeneous_transaction_graph(val_df, feat_cols)
        with torch.no_grad():
            emb_val = gnn(graph_val.x, graph_val.edge_index).numpy()
        raw_val = model.predict_proba(emb_val)
        graph_test = build_homogeneous_transaction_graph(test_df, feat_cols)
        with torch.no_grad():
            emb_test = gnn(graph_test.x, graph_test.edge_index).numpy()
        raw_test = model.predict_proba(emb_test)
        # Align lengths
        raw_val = raw_val[: len(y_val)]
        raw_test = raw_test[: len(y_test)]

    # Calibration
    calibrator = Calibrator(CalibrationMethod.ISOTONIC)
    calibrator.fit(raw_val, y_val)
    cal_val = calibrator.transform(raw_val)
    cal_test = calibrator.transform(raw_test)

    result["metrics"] = {
        "pr_auc": pr_auc(y_test, raw_test),
        "recall_at_1pct_fpr": recall_at_fpr(y_test, raw_test, 0.01),
        "brier_raw": brier(y_test, raw_test),
        "brier_calibrated": brier(y_test, cal_test),
        "ece_raw": ece(y_test, raw_test),
        "ece_calibrated": ece(y_test, cal_test),
    }

    decisions, _ = conformal_triage(cal_test, fraud_threshold=0.5, gap=0.2)
    result["metrics"]["review_rate"] = (decisions == 1).mean()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save calibrator and model refs (simplified - full impl would pickle/save)
        save_json(
            {"model_type": model_type, "metrics": result["metrics"], "feat_cols": feat_cols},
            output_dir / "train_result.json",
        )

    result["model"] = model
    result["calibrator"] = calibrator
    result["feat_cols"] = feat_cols
    return result
