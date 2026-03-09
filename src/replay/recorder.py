"""Decision recorder: stores every prediction in DuckDB for audit replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from replay.hashing import decision_hash
from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    tx_id VARCHAR PRIMARY KEY,
    payload JSON,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS features (
    tx_id VARCHAR PRIMARY KEY,
    feature_vector JSON,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    decision_id VARCHAR PRIMARY KEY,
    tx_id VARCHAR,
    raw_score DOUBLE,
    calibrated_score DOUBLE,
    confidence_band VARCHAR,
    interval_low DOUBLE,
    interval_high DOUBLE,
    model_id VARCHAR,
    model_version VARCHAR,
    calibration_version VARCHAR,
    conformal_version VARCHAR,
    explanation TEXT,
    decision_hash VARCHAR,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS model_registry (
    model_id VARCHAR PRIMARY KEY,
    model_type VARCHAR,
    version VARCHAR,
    artifact_path VARCHAR,
    metrics JSON,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit_reports (
    decision_id VARCHAR PRIMARY KEY,
    report_markdown TEXT,
    report_json JSON,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS replay_events (
    replay_id VARCHAR PRIMARY KEY,
    decision_id VARCHAR,
    matched BOOLEAN,
    diff JSON,
    replayed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class DecisionRecorder:
    """Records predictions and their context into DuckDB for audit trail."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or cfg.audit_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        for stmt in SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    def record_transaction(self, tx_id: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO transactions (tx_id, payload) VALUES (?, ?)",
            [tx_id, json.dumps(payload, default=str)],
        )

    def record_features(self, tx_id: str, features: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO features (tx_id, feature_vector) VALUES (?, ?)",
            [tx_id, json.dumps(features)],
        )

    def record_prediction(self, prediction: dict[str, Any]) -> str:
        hashable_payload = {
            "tx_id": prediction["tx_id"],
            "raw_score": prediction["raw_score"],
            "calibrated_score": prediction["calibrated_score"],
            "confidence_band": prediction["confidence_band"],
            "model_type": prediction.get("model_type", ""),
        }
        d_hash = decision_hash(hashable_payload)

        self.conn.execute(
            """INSERT OR REPLACE INTO predictions
            (decision_id, tx_id, raw_score, calibrated_score, confidence_band,
             interval_low, interval_high, model_id, model_version,
             calibration_version, conformal_version, explanation, decision_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                prediction["decision_id"],
                prediction["tx_id"],
                prediction["raw_score"],
                prediction["calibrated_score"],
                prediction["confidence_band"],
                prediction.get("interval_low", 0),
                prediction.get("interval_high", 1),
                prediction.get("model_type", ""),
                prediction.get("model_version", "1.0"),
                prediction.get("calibration_version", "1.0"),
                prediction.get("conformal_version", "1.0"),
                prediction.get("explanation", ""),
                d_hash,
            ],
        )
        log.info("prediction_recorded", decision_id=prediction["decision_id"])
        return d_hash

    def record_model(self, model_info: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO model_registry
            (model_id, model_type, version, artifact_path, metrics)
            VALUES (?, ?, ?, ?, ?)""",
            [
                model_info["model_id"],
                model_info["model_type"],
                model_info.get("version", "1.0"),
                model_info.get("artifact_path", ""),
                json.dumps(model_info.get("metrics", {})),
            ],
        )

    def record_audit_report(self, decision_id: str, markdown: str, report_json: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO audit_reports
            (decision_id, report_markdown, report_json) VALUES (?, ?, ?)""",
            [decision_id, markdown, json.dumps(report_json, default=str)],
        )

    def get_prediction(self, decision_id: str) -> dict | None:
        result = self.conn.execute(
            "SELECT * FROM predictions WHERE decision_id = ?", [decision_id]
        ).fetchone()
        if result is None:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def get_transaction(self, tx_id: str) -> dict | None:
        result = self.conn.execute(
            "SELECT * FROM transactions WHERE tx_id = ?", [tx_id]
        ).fetchone()
        if result is None:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def get_features(self, tx_id: str) -> dict | None:
        result = self.conn.execute(
            "SELECT * FROM features WHERE tx_id = ?", [tx_id]
        ).fetchone()
        if result is None:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def list_decisions(self, limit: int = 100) -> list[dict]:
        results = self.conn.execute(
            "SELECT decision_id, tx_id, confidence_band, recorded_at FROM predictions ORDER BY recorded_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        cols = ["decision_id", "tx_id", "confidence_band", "recorded_at"]
        return [dict(zip(cols, row)) for row in results]

    def close(self):
        self.conn.close()
