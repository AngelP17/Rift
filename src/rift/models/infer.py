"""Inference pipeline for Rift."""

from typing import Any, Optional

import numpy as np
import polars as pl
import torch

from rift.data.schemas import FEATURE_COLUMNS
from rift.features.engine import compute_features
from rift.graph.builder import build_homogeneous_transaction_graph
from rift.models.calibrate import Calibrator
from rift.models.conformal import DecisionBand, conformal_triage


def predict(
    tx: dict | pl.DataFrame,
    model: Any,
    calibrator: Optional[Calibrator] = None,
    feat_cols: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Single transaction prediction.

    Returns: raw_score, calibrated_score, decision_band, confidence
    """
    if isinstance(tx, dict):
        df = pl.DataFrame([tx])
    else:
        df = tx
    feat_cols = feat_cols or [c for c in FEATURE_COLUMNS if c in df.columns]
    df_feat = compute_features(df)
    X = df_feat.select(feat_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)

    if hasattr(model, "predict_proba") and not hasattr(model, "get_embeddings"):
        raw = model.predict_proba(X)[0]
    else:
        graph = build_homogeneous_transaction_graph(df_feat, feat_cols)
        raw = model.predict_proba(graph.x, graph.edge_index, X)[0]

    cal = raw
    if calibrator:
        cal = calibrator.transform(np.array([raw]))[0]

    decisions, conf = conformal_triage(np.array([cal]), fraud_threshold=0.5, gap=0.2)
    band = DecisionBand.REVIEW_NEEDED
    if decisions[0] == 0:
        band = DecisionBand.HIGH_CONFIDENCE_FRAUD
    elif decisions[0] == 2:
        band = DecisionBand.HIGH_CONFIDENCE_LEGIT

    return {
        "raw_score": float(raw),
        "calibrated_score": float(cal),
        "decision_band": band.value,
        "confidence": float(conf[0]),
        "is_fraud_pred": cal > 0.5,
    }
