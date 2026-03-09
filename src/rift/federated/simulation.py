from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from rift.data.splits import chronological_split, random_split
from rift.features.engine import build_features, extract_categorical_mappings, feature_columns
from rift.mlops.mlflow_tracker import log_run_metrics
from rift.models.calibrate import ProbabilityCalibrator
from rift.models.conformal import ConformalClassifier
from rift.models.metrics import brier, expected_calibration_error, pr_auc, recall_at_fpr
from rift.optimize.green import apply_green_optimization
from rift.utils.config import RiftPaths
from rift.utils.io import read_json, write_json, write_pickle


@dataclass(frozen=True)
class FederatedRunSummary:
    run_id: str
    client_column: str
    rounds: int
    local_epochs: int
    client_count: int
    metrics: dict[str, float]
    artifact_path: str
    metadata_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FederatedLogisticModel:
    def __init__(self, weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> None:
        self.weights = weights.astype(float)
        self.bias = float(bias)
        self.mean = mean.astype(float)
        self.std = std.astype(float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.transform(x) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _select(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select(columns).to_numpy().astype(float)


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def _local_train(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    local_epochs: int,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    local_w = weights.copy()
    local_b = float(bias)
    for _ in range(local_epochs):
        logits = x @ local_w + local_b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = probs - y
        grad_w = (x.T @ error) / max(len(y), 1) + l2 * local_w
        grad_b = float(error.mean()) if len(y) else 0.0
        local_w -= learning_rate * grad_w
        local_b -= learning_rate * grad_b
    return local_w, local_b


def _split(frame: pl.DataFrame, time_split: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    split = chronological_split(frame) if time_split else random_split(frame)
    return split.train, split.validation, split.test


def train_federated_model(
    frame: pl.DataFrame,
    paths: RiftPaths,
    client_column: str = "channel",
    rounds: int = 5,
    local_epochs: int = 3,
    learning_rate: float = 0.1,
    time_split: bool = False,
    optimize_mode: str = "standard",
) -> FederatedRunSummary:
    feat = build_features(frame)
    categorical_mappings = extract_categorical_mappings(feat)
    train_frame, val_frame, test_frame = _split(feat, time_split=time_split)
    if client_column not in train_frame.columns:
        raise ValueError(f"client column '{client_column}' not found in training frame")

    columns = feature_columns(feat)
    train_x = _select(train_frame, columns)
    val_x = _select(val_frame, columns)
    test_x = _select(test_frame, columns)
    train_y = train_frame["is_fraud"].to_numpy().astype(float)
    val_y = val_frame["is_fraud"].to_numpy().astype(int)
    test_y = test_frame["is_fraud"].to_numpy().astype(int)

    mean, std = _fit_scaler(train_x)
    train_x_scaled = (train_x - mean) / std
    val_x_scaled = (val_x - mean) / std
    test_x_scaled = (test_x - mean) / std

    client_values = [str(value) for value in train_frame[client_column].to_list()]
    unique_clients = sorted(set(client_values))
    weights = np.zeros(train_x.shape[1], dtype=float)
    bias = 0.0
    client_stats: list[dict[str, Any]] = []

    for round_idx in range(rounds):
        local_models: list[tuple[np.ndarray, float, int]] = []
        client_stats = []
        for client in unique_clients:
            mask = np.asarray([value == client for value in client_values], dtype=bool)
            x_client = train_x_scaled[mask]
            y_client = train_y[mask]
            if x_client.size == 0:
                continue
            local_w, local_b = _local_train(
                x_client,
                y_client,
                weights=weights,
                bias=bias,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
            )
            local_models.append((local_w, local_b, x_client.shape[0]))
            client_stats.append({"client": client, "samples": int(x_client.shape[0]), "round": round_idx + 1})

        if not local_models:
            raise ValueError("no federated clients available for training")

        total_samples = sum(samples for _, _, samples in local_models)
        weights = sum(local_w * (samples / total_samples) for local_w, _, samples in local_models)
        bias = sum(local_b * (samples / total_samples) for _, local_b, samples in local_models)

    model = FederatedLogisticModel(weights=weights, bias=bias, mean=mean, std=std)
    val_raw = model.predict_proba(val_x)
    test_raw = model.predict_proba(test_x)
    calibrator = ProbabilityCalibrator(method="isotonic")
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

    run_id = f"fed_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    run_dir = paths.federated_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / "artifact.pkl"
    metadata_path = run_dir / "metrics.json"
    artifact = {
        "model_type": "federated_logistic",
        "client_column": client_column,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "feature_columns": columns,
        "categorical_mappings": categorical_mappings,
        "model": model,
        "calibrator": calibrator,
        "conformal": conformal,
        "client_stats": client_stats,
    }
    artifact, optimization = apply_green_optimization(artifact, optimize_mode)
    write_pickle(artifact_path, artifact)
    write_json(
        metadata_path,
        {
            "run_id": run_id,
            "client_column": client_column,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "client_count": len(unique_clients),
            "metrics": metrics,
            "artifact_path": str(artifact_path),
            "optimization": optimization,
        },
    )
    mlflow_run_id = log_run_metrics(
        tracking_dir=paths.mlflow_dir,
        experiment_name="rift-federated",
        run_name=run_id,
        params={
            "client_column": client_column,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "time_split": time_split,
            "optimize_mode": optimize_mode,
        },
        metrics=metrics,
        tags={"component": "federated"},
    )
    if mlflow_run_id is not None:
        write_json(
            metadata_path,
            {
                "run_id": run_id,
                "client_column": client_column,
                "rounds": rounds,
                "local_epochs": local_epochs,
                "client_count": len(unique_clients),
                "metrics": metrics,
                "artifact_path": str(artifact_path),
                "optimization": optimization,
                "mlflow_run_id": mlflow_run_id,
            },
        )
    write_json(paths.federated_dir / "current_federated_run.json", {"run_id": run_id, "artifact_path": str(artifact_path)})
    return FederatedRunSummary(
        run_id=run_id,
        client_column=client_column,
        rounds=rounds,
        local_epochs=local_epochs,
        client_count=len(unique_clients),
        metrics=metrics,
        artifact_path=str(artifact_path),
        metadata_path=str(metadata_path),
    )


def list_federated_runs(paths: RiftPaths, limit: int = 10) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metrics_file in sorted(paths.federated_dir.glob("fed_*/metrics.json"), reverse=True)[:limit]:
        output.append(read_json(metrics_file))
    return output
