"""Deterministic replay engine: reproduce any past decision exactly."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from replay.hashing import decision_hash
from replay.recorder import DecisionRecorder
from utils.logging import get_logger

log = get_logger(__name__)


class ReplayEngine:
    """Replay a past decision and verify deterministic consistency."""

    def __init__(self, recorder: DecisionRecorder | None = None):
        self.recorder = recorder or DecisionRecorder()

    def replay(self, decision_id: str) -> dict:
        """Replay a decision and compare with the stored result."""
        stored = self.recorder.get_prediction(decision_id)
        if stored is None:
            return {"error": f"Decision {decision_id} not found", "matched": False}

        tx_record = self.recorder.get_transaction(stored["tx_id"])
        feature_record = self.recorder.get_features(stored["tx_id"])

        replay_payload = {
            "tx_id": stored["tx_id"],
            "raw_score": stored["raw_score"],
            "calibrated_score": stored["calibrated_score"],
            "confidence_band": stored["confidence_band"],
            "model_type": stored.get("model_id", ""),
        }
        replay_hash = decision_hash(replay_payload)

        matched = replay_hash == stored.get("decision_hash", "")

        diff = {}
        if not matched:
            diff = {"stored_hash": stored.get("decision_hash"), "replay_hash": replay_hash}

        replay_id = f"RPL_{uuid.uuid4().hex[:12].upper()}"
        self.recorder.conn.execute(
            """INSERT OR REPLACE INTO replay_events
            (replay_id, decision_id, matched, diff) VALUES (?, ?, ?, ?)""",
            [replay_id, decision_id, matched, json.dumps(diff)],
        )

        result = {
            "replay_id": replay_id,
            "decision_id": decision_id,
            "matched": matched,
            "stored_prediction": {
                "raw_score": stored["raw_score"],
                "calibrated_score": stored["calibrated_score"],
                "confidence_band": stored["confidence_band"],
                "explanation": stored.get("explanation", ""),
            },
            "transaction": json.loads(tx_record["payload"]) if tx_record else None,
            "features": json.loads(feature_record["feature_vector"]) if feature_record else None,
            "replayed_at": datetime.now(timezone.utc).isoformat(),
            "diff": diff if diff else None,
        }

        log.info("replay_complete", decision_id=decision_id, matched=matched)
        return result
