"""Inference pipeline: load artifacts and predict on new transactions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from features.engine import FEATURE_COLUMNS, build_features, get_feature_matrix
from graph.builder import build_graph
from graph.hetero_graph import to_homogeneous_projection
from models.baseline_xgb import TabularXGBoost
from models.calibrate import IsotonicCalibrator, PlattCalibrator
from models.conformal import ConformalPredictor
from models.ensemble import HybridEnsemble
from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class InferencePipeline:
    """End-to-end inference: features -> model -> calibrate -> conformal -> explain."""

    def __init__(
        self,
        model_type: str = "graphsage_xgb",
        model_path: Path | None = None,
        calibrator_path: Path | None = None,
        conformal_path: Path | None = None,
    ):
        self.model_type = model_type
        self.model = self._load_model(model_type, model_path)
        self.calibrator = self._load_calibrator(calibrator_path)
        self.conformal = ConformalPredictor.load(conformal_path)

    def _load_model(self, model_type: str, path: Path | None):
        if model_type == "xgb_tabular":
            return TabularXGBoost.load(path)
        else:
            return HybridEnsemble.load(path or (cfg.model_dir / "hybrid_graphsage_xgboost.pkl"))

    def _load_calibrator(self, path: Path | None):
        path = path or (cfg.model_dir / "calibrator_isotonic.pkl")
        if path.exists():
            return IsotonicCalibrator.load(path)
        return PlattCalibrator.load(cfg.model_dir / "calibrator_platt.pkl")

    def predict(self, tx_data: dict | pl.DataFrame) -> dict:
        """Run full inference on a transaction."""
        if isinstance(tx_data, dict):
            tx_data = pl.DataFrame([tx_data])

        tx_data = build_features(tx_data)
        features = get_feature_matrix(tx_data).to_numpy().astype(np.float32)

        if self.model_type == "xgb_tabular":
            raw_score = float(self.model.predict_proba(features)[0])
        else:
            g = build_graph(tx_data, FEATURE_COLUMNS)
            x, edge_index, _ = to_homogeneous_projection(g)
            raw_score = float(self.model.predict_proba(x, edge_index, features)[0])

        calibrated = float(self.calibrator.calibrate(np.array([raw_score]))[0])
        conformal_result = self.conformal.predict(np.array([calibrated]))[0]

        decision_id = f"DEC_{uuid.uuid4().hex[:16].upper()}"

        return {
            "decision_id": decision_id,
            "tx_id": tx_data["tx_id"][0],
            "raw_score": raw_score,
            "calibrated_score": calibrated,
            "confidence_band": conformal_result["confidence_band"],
            "interval_low": conformal_result["interval_low"],
            "interval_high": conformal_result["interval_high"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": self.model_type,
            "features": features[0].tolist(),
        }


def predict_single(tx_json: str | dict, model_type: str = "graphsage_xgb") -> dict:
    """Convenience function for single-transaction prediction."""
    if isinstance(tx_json, str):
        tx_data = json.loads(tx_json)
    else:
        tx_data = tx_json
    pipeline = InferencePipeline(model_type=model_type)
    return pipeline.predict(tx_data)
