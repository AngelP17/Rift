"""Training pipeline orchestration for all Rift models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from data.splits import split_data
from features.engine import FEATURE_COLUMNS, build_features, get_feature_matrix
from graph.builder import build_graph
from graph.hetero_graph import to_homogeneous_projection
from models.baseline_xgb import TabularXGBoost
from models.calibrate import calibrate_scores
from models.conformal import ConformalPredictor
from models.ensemble import HybridEnsemble
from models.gat import GATClassifier
from models.graphsage import GraphSAGEClassifier
from models.metrics import compute_all_metrics
from utils.config import cfg
from utils.logging import get_logger
from utils.seeds import set_global_seeds

log = get_logger(__name__)

ModelType = Literal["xgb_tabular", "graphsage_only", "graphsage_xgb", "gat_xgb"]


def train_pipeline(
    df,
    model_type: ModelType = "graphsage_xgb",
    split_strategy: str = "temporal",
    window_days: int = 7,
    calibration_method: str = "isotonic",
    seed: int = 42,
    epochs: int = 50,
    booster: str = "xgboost",
) -> dict:
    """Full training pipeline: features -> split -> train -> calibrate -> conformal."""
    set_global_seeds(seed)
    cfg.ensure_dirs()

    log.info("starting_training", model_type=model_type, split=split_strategy)

    df_feat = build_features(df)
    splits = split_data(df_feat, strategy=split_strategy, window_days=window_days, seed=seed)

    train_df = splits["train"]
    val_df = splits.get("val", splits.get("test_fold_0", train_df[:100]))
    test_df = splits.get("test", splits.get("test_fold_0", train_df[:100]))

    if model_type == "xgb_tabular":
        result = _train_xgb_tabular(train_df, val_df, test_df, calibration_method)
    elif model_type == "graphsage_only":
        result = _train_graphsage_only(train_df, val_df, test_df, calibration_method, epochs)
    elif model_type == "graphsage_xgb":
        result = _train_hybrid(train_df, val_df, test_df, calibration_method, epochs, booster, "graphsage")
    elif model_type == "gat_xgb":
        result = _train_hybrid(train_df, val_df, test_df, calibration_method, epochs, booster, "gat")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result["model_type"] = model_type
    result["trained_at"] = datetime.now(timezone.utc).isoformat()
    result["split_strategy"] = split_strategy
    return result


def _train_xgb_tabular(train_df, val_df, test_df, cal_method) -> dict:
    X_train = get_feature_matrix(train_df).to_numpy().astype(np.float32)
    y_train = train_df["is_fraud"].to_numpy().astype(np.float32)
    X_val = get_feature_matrix(val_df).to_numpy().astype(np.float32)
    y_val = val_df["is_fraud"].to_numpy().astype(np.float32)
    X_test = get_feature_matrix(test_df).to_numpy().astype(np.float32)
    y_test = test_df["is_fraud"].to_numpy().astype(np.float32)

    model = TabularXGBoost()
    model.fit(X_train, y_train, X_val, y_val)

    raw_scores = model.predict_proba(X_test)
    raw_metrics = compute_all_metrics(y_test, raw_scores, prefix="raw")

    cal_scores_val = model.predict_proba(X_val)
    cal_scores, calibrator = calibrate_scores(cal_scores_val, y_val, method=cal_method)
    calibrated_test = calibrator.calibrate(raw_scores)
    cal_metrics = compute_all_metrics(y_test, calibrated_test, prefix="calibrated")

    cp = ConformalPredictor().fit(calibrated_test, y_test)

    model.save()
    calibrator.save()
    cp.save()

    return {
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "model_path": str(cfg.model_dir / "xgb_tabular.pkl"),
    }


def _train_gnn(model, x, edge_index, labels, mask_train, mask_val, epochs, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[mask_train], labels[mask_train].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_loss = criterion(val_out[mask_val], labels[mask_val].float())

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("early_stopping", epoch=epoch)
                break

    return model


def _train_graphsage_only(train_df, val_df, test_df, cal_method, epochs) -> dict:
    combined = _concat_dfs(train_df, val_df, test_df)
    n_train = len(train_df)
    n_val = len(val_df)

    g = build_graph(combined, FEATURE_COLUMNS)
    x, edge_index, labels = to_homogeneous_projection(g, "transaction")

    mask_train = torch.zeros(len(combined), dtype=torch.bool)
    mask_train[:n_train] = True
    mask_val = torch.zeros(len(combined), dtype=torch.bool)
    mask_val[n_train:n_train + n_val] = True
    mask_test = torch.zeros(len(combined), dtype=torch.bool)
    mask_test[n_train + n_val:] = True

    model = GraphSAGEClassifier(in_channels=x.size(1))
    model = _train_gnn(model, x, edge_index, labels, mask_train, mask_val, epochs)

    model.eval()
    with torch.no_grad():
        raw_scores = torch.sigmoid(model(x, edge_index)).numpy()

    test_scores = raw_scores[mask_test.numpy()]
    test_labels = labels[mask_test].numpy()
    raw_metrics = compute_all_metrics(test_labels, test_scores, prefix="raw")

    val_scores = raw_scores[mask_val.numpy()]
    val_labels = labels[mask_val].numpy()
    _, calibrator = calibrate_scores(val_scores, val_labels, method=cal_method)
    calibrated_test = calibrator.calibrate(test_scores)
    cal_metrics = compute_all_metrics(test_labels, calibrated_test, prefix="calibrated")

    cp = ConformalPredictor().fit(calibrated_test, test_labels)

    model.save()
    calibrator.save()
    cp.save()

    return {
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "model_path": str(cfg.model_dir / "graphsage.pt"),
    }


def _train_hybrid(train_df, val_df, test_df, cal_method, epochs, booster, encoder_type) -> dict:
    combined = _concat_dfs(train_df, val_df, test_df)
    n_train = len(train_df)
    n_val = len(val_df)

    g = build_graph(combined, FEATURE_COLUMNS)
    x, edge_index, labels = to_homogeneous_projection(g, "transaction")

    mask_train = torch.zeros(len(combined), dtype=torch.bool)
    mask_train[:n_train] = True
    mask_val = torch.zeros(len(combined), dtype=torch.bool)
    mask_val[n_train:n_train + n_val] = True
    mask_test = torch.zeros(len(combined), dtype=torch.bool)
    mask_test[n_train + n_val:] = True

    if encoder_type == "gat":
        gnn = GATClassifier(in_channels=x.size(1))
    else:
        gnn = GraphSAGEClassifier(in_channels=x.size(1))

    gnn = _train_gnn(gnn, x, edge_index, labels, mask_train, mask_val, epochs)

    tabular_train = get_feature_matrix(train_df).to_numpy().astype(np.float32)
    tabular_val = get_feature_matrix(val_df).to_numpy().astype(np.float32)
    tabular_test = get_feature_matrix(test_df).to_numpy().astype(np.float32)

    ensemble = HybridEnsemble(gnn, booster=booster)

    x[:n_train]
    _filter_edges(edge_index, n_train)

    x[n_train:n_train + n_val]
    _filter_edges(edge_index, n_val, offset=n_train)

    labels[:n_train].numpy().astype(np.float32)
    y_val = labels[n_train:n_train + n_val].numpy().astype(np.float32)
    y_test = labels[n_train + n_val:].numpy().astype(np.float32)

    ensemble.fit(
        x[:n_train + n_val + len(test_df)], edge_index,
        labels[:n_train].numpy().astype(np.float32),
        tabular_train,
    )

    raw_scores = ensemble.predict_proba(
        x[:n_train + n_val + len(test_df)], edge_index, tabular_test,
    )[:len(test_df)]
    raw_metrics = compute_all_metrics(y_test, raw_scores, prefix="raw")

    val_raw = ensemble.predict_proba(
        x[:n_train + n_val + len(test_df)], edge_index, tabular_val,
    )[:len(val_df)]
    _, calibrator = calibrate_scores(val_raw, y_val, method=cal_method)
    calibrated_test = calibrator.calibrate(raw_scores)
    cal_metrics = compute_all_metrics(y_test, calibrated_test, prefix="calibrated")

    cp = ConformalPredictor().fit(calibrated_test, y_test)

    name = f"hybrid_{encoder_type}_{booster}"
    gnn.save(cfg.model_dir / f"{encoder_type}.pt")
    ensemble.save(cfg.model_dir / f"{name}.pkl")
    calibrator.save()
    cp.save()

    return {
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "model_path": str(cfg.model_dir / f"{name}.pkl"),
    }


def _filter_edges(edge_index: torch.Tensor, n: int, offset: int = 0) -> torch.Tensor:
    mask = (edge_index[0] < n + offset) & (edge_index[1] < n + offset)
    return edge_index[:, mask]


def _concat_dfs(train_df, val_df, test_df):
    import polars as pl
    return pl.concat([train_df, val_df, test_df])
