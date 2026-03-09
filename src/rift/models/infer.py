from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from rift.features.engine import build_features
from rift.graph.builder import build_transaction_graph
from rift.models.train import _select
from rift.utils.io import read_json, read_pickle


def load_run(runs_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    if run_id is None:
        current = read_json(runs_dir / "current_run.json")
        artifact_path = Path(current["artifact_path"])
        artifact = read_pickle(artifact_path)
        artifact["_run_id"] = current["run_id"]
        return artifact
    artifact = read_pickle(runs_dir / run_id / "artifact.pkl")
    artifact["_run_id"] = run_id
    return artifact


def payload_to_frame(payload: dict[str, Any]) -> pl.DataFrame:
    parsed = dict(payload)
    timestamp = parsed.get("timestamp")
    if isinstance(timestamp, str):
        parsed["timestamp"] = datetime.fromisoformat(timestamp)
    parsed.setdefault("currency", "USD")
    parsed.setdefault("is_fraud", 0)
    return pl.DataFrame([parsed])


def score_frame(frame: pl.DataFrame, artifact: dict[str, Any]) -> tuple[dict[str, Any], pl.DataFrame]:
    feature_frame = build_features(frame, categorical_mappings=artifact.get("categorical_mappings"))
    columns = artifact["feature_columns"]
    matrix = _select(feature_frame, columns)
    model_type = artifact["model_type"]
    if model_type == "xgb_tabular":
        raw = artifact["model"].predict_proba(matrix)
    else:
        graph = build_transaction_graph(feature_frame)
        raw = artifact["model"].predict_proba(matrix, graph)
    calibrated = artifact["calibrator"].predict(raw)
    decision, confidence = artifact["conformal"].triage(calibrated)[0]
    return (
        {
            "raw_probability": float(raw[0]),
            "calibrated_probability": float(calibrated[0]),
            "decision": decision,
            "confidence": float(confidence),
            "model_type": model_type,
        },
        feature_frame,
    )
