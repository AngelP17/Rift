"""Tests for the replay and recording system."""

import json

from replay.hashing import canonical_json, decision_hash, verify_hash
from replay.recorder import DecisionRecorder
from replay.replayer import ReplayEngine


class TestHashing:
    def test_canonical_json_deterministic(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert canonical_json(d1) == canonical_json(d2)

    def test_decision_hash_deterministic(self):
        payload = {"tx_id": "TX001", "score": 0.85}
        h1 = decision_hash(payload)
        h2 = decision_hash(payload)
        assert h1 == h2
        assert len(h1) == 64

    def test_verify_hash(self):
        payload = {"tx_id": "TX001", "score": 0.85}
        h = decision_hash(payload)
        assert verify_hash(payload, h)
        assert not verify_hash({"tx_id": "TX002"}, h)


class TestRecorder:
    def test_record_and_retrieve_prediction(self, tmp_db):
        recorder = DecisionRecorder(tmp_db)
        prediction = {
            "decision_id": "DEC_TEST001",
            "tx_id": "TX_001",
            "raw_score": 0.85,
            "calibrated_score": 0.82,
            "confidence_band": "high_confidence_fraud",
            "model_type": "graphsage_xgb",
        }
        recorder.record_prediction(prediction)
        retrieved = recorder.get_prediction("DEC_TEST001")
        assert retrieved is not None
        assert retrieved["tx_id"] == "TX_001"
        assert abs(retrieved["raw_score"] - 0.85) < 1e-6
        recorder.close()

    def test_record_transaction(self, tmp_db):
        recorder = DecisionRecorder(tmp_db)
        recorder.record_transaction("TX_001", {"amount": 100, "user": "U1"})
        tx = recorder.get_transaction("TX_001")
        assert tx is not None
        payload = json.loads(tx["payload"])
        assert payload["amount"] == 100
        recorder.close()

    def test_list_decisions(self, tmp_db):
        recorder = DecisionRecorder(tmp_db)
        for i in range(5):
            recorder.record_prediction({
                "decision_id": f"DEC_{i:03d}",
                "tx_id": f"TX_{i:03d}",
                "raw_score": 0.5,
                "calibrated_score": 0.5,
                "confidence_band": "review_needed",
                "model_type": "test",
            })
        decisions = recorder.list_decisions()
        assert len(decisions) == 5
        recorder.close()


class TestReplayEngine:
    def test_replay_existing(self, tmp_db):
        recorder = DecisionRecorder(tmp_db)
        recorder.record_transaction("TX_R001", {"amount": 200})
        recorder.record_features("TX_R001", [1.0, 2.0, 3.0])
        recorder.record_prediction({
            "decision_id": "DEC_R001",
            "tx_id": "TX_R001",
            "raw_score": 0.75,
            "calibrated_score": 0.72,
            "confidence_band": "review_needed",
            "model_type": "xgb",
        })

        engine = ReplayEngine(recorder)
        result = engine.replay("DEC_R001")

        assert result["matched"]
        assert result["stored_prediction"]["raw_score"] == 0.75
        recorder.close()

    def test_replay_nonexistent(self, tmp_db):
        engine = ReplayEngine(DecisionRecorder(tmp_db))
        result = engine.replay("DEC_NONEXISTENT")
        assert not result["matched"]
        assert "error" in result
