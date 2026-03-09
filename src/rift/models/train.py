from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from rift.data.splits import DatasetSplit, chronological_split, random_split
from rift.features.engine import build_features, extract_categorical_mappings, feature_columns
from rift.graph.builder import build_transaction_graph
from rift.models.baseline_xgb import TabularXGBoostModel
from rift.models.calibrate import ProbabilityCalibrator
from rift.models.conformal import ConformalClassifier
from rift.models.ensemble import GraphHybridModel
from rift.models.graphsage import GraphSAGEEncoder, GraphSAGEOnlyModel
from rift.models.metrics import brier, expected_calibration_error, pr_auc, recall_at_fpr
from rift.utils.io import write_json, write_pickle


@dataclass
class ModelRunSummary:
    run_id: str
    model_type: str
    metrics: dict[str, float]
    artifact_path: str
    metadata_path: str


def _select(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select(columns).to_numpy().astype(float)


def _labels(frame: pl.DataFrame) -> np.ndarray:
    return frame["is_fraud"].to_numpy().astype(int)


def _split_frame(frame: pl.DataFrame, time_split: bool) -> DatasetSplit:
    return chronological_split(frame) if time_split else random_split(frame)


def train_from_frame(
    frame: pl.DataFrame,
    runs_dir: Path,
    model_type: str = "graphsage_xgb",
    time_split: bool = False,
    calibration_method: str = "isotonic",
) -> ModelRunSummary:
    feat = build_features(frame)
    categorical_mappings = extract_categorical_mappings(feat)
    split = _split_frame(feat, time_split=time_split)
    columns = feature_columns(feat)
    train_x = _select(split.train, columns)
    val_x = _select(split.validation, columns)
    test_x = _select(split.test, columns)
    train_y = _labels(split.train)
    val_y = _labels(split.validation)
    test_y = _labels(split.test)

    train_graph = build_transaction_graph(split.train)
    val_graph = build_transaction_graph(split.validation)
    test_graph = build_transaction_graph(split.test)

    artifact: dict[str, Any] = {
        "model_type": model_type,
        "feature_columns": columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    if model_type == "xgb_tabular":
        model = TabularXGBoostModel()
        model.fit(train_x, train_y)
        val_raw = model.predict_proba(val_x)
        test_raw = model.predict_proba(test_x)
        artifact["model"] = model
    elif model_type == "graphsage_only":
        encoder = GraphSAGEEncoder(in_dim=train_x.shape[1])
        model = GraphSAGEOnlyModel(encoder)
        model.fit(train_x, train_graph, train_y)
        val_raw = model.predict_proba(val_x, val_graph)
        test_raw = model.predict_proba(test_x, test_graph)
        artifact["model"] = model
    else:
        encoder = GraphSAGEEncoder(in_dim=train_x.shape[1])
        model = GraphHybridModel(encoder)
        model.fit(train_x, train_graph, train_y)
        val_raw = model.predict_proba(val_x, val_graph)
        test_raw = model.predict_proba(test_x, test_graph)
        artifact["model"] = model

    calibrator = ProbabilityCalibrator(method=calibration_method)
    calibrator.fit(val_raw, val_y)
    test_calibrated = calibrator.predict(test_raw)

    conformal = ConformalClassifier(alpha=0.05)
    conformal.fit(calibrator.predict(val_raw), val_y)
    decisions = conformal.triage(test_calibrated)

    metrics = {
        "pr_auc": pr_auc(test_y, test_calibrated),
        "recall_at_1pct_fpr": recall_at_fpr(test_y, test_calibrated, target_fpr=0.01),
        "brier": brier(test_y, test_calibrated),
        "ece": expected_calibration_error(test_y, test_calibrated),
        "review_rate": float(sum(decision == "review_needed" for decision, _ in decisions) / max(len(decisions), 1)),
    }
    artifact["calibrator"] = calibrator
    artifact["conformal"] = conformal
    artifact["categorical_mappings"] = categorical_mappings
    artifact["train_reference"] = {
        "features": train_x[:500].tolist(),
        "labels": train_y[:500].tolist(),
        "tx_ids": split.train["tx_id"].to_list()[:500],
    }

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    run_dir = runs_dir / run_id
    artifact_path = run_dir / "artifact.pkl"
    metadata_path = run_dir / "metrics.json"
    write_pickle(artifact_path, artifact)
    write_json(
        metadata_path,
        {
            "run_id": run_id,
            "model_type": model_type,
            "metrics": metrics,
            "feature_columns": columns,
            "time_split": time_split,
            "calibration_method": calibration_method,
        },
    )
    write_json(runs_dir / "current_run.json", {"run_id": run_id, "artifact_path": str(artifact_path)})
    return ModelRunSummary(
        run_id=run_id,
        model_type=model_type,
        metrics=metrics,
        artifact_path=str(artifact_path),
        metadata_path=str(metadata_path),
    )
