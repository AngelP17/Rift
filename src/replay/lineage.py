"""Decision lineage tracking: trace a decision back to its data, model, and artifacts."""

from __future__ import annotations

import json

from replay.recorder import DecisionRecorder
from utils.logging import get_logger

log = get_logger(__name__)


class LineageTracker:
    """Provides full provenance chain for any decision."""

    def __init__(self, recorder: DecisionRecorder | None = None):
        self.recorder = recorder or DecisionRecorder()

    def get_lineage(self, decision_id: str) -> dict:
        """Return full lineage for a decision."""
        pred = self.recorder.get_prediction(decision_id)
        if pred is None:
            return {"error": f"Decision {decision_id} not found"}

        tx = self.recorder.get_transaction(pred["tx_id"])
        features = self.recorder.get_features(pred["tx_id"])

        report = self.recorder.conn.execute(
            "SELECT * FROM audit_reports WHERE decision_id = ?", [decision_id]
        ).fetchone()

        replays = self.recorder.conn.execute(
            "SELECT replay_id, matched, replayed_at FROM replay_events WHERE decision_id = ? ORDER BY replayed_at DESC",
            [decision_id],
        ).fetchall()

        return {
            "decision_id": decision_id,
            "prediction": {
                "raw_score": pred["raw_score"],
                "calibrated_score": pred["calibrated_score"],
                "confidence_band": pred["confidence_band"],
                "decision_hash": pred.get("decision_hash"),
                "recorded_at": str(pred.get("recorded_at", "")),
            },
            "transaction": json.loads(tx["payload"]) if tx else None,
            "features": json.loads(features["feature_vector"]) if features else None,
            "model": {
                "model_id": pred.get("model_id"),
                "model_version": pred.get("model_version"),
                "calibration_version": pred.get("calibration_version"),
                "conformal_version": pred.get("conformal_version"),
            },
            "audit_report": report is not None,
            "replay_history": [
                {"replay_id": r[0], "matched": r[1], "replayed_at": str(r[2])}
                for r in replays
            ],
        }
