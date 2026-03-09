"""Deterministic decision recorder for audit trail."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

from rift.replay.hashing import canonical_hash
from rift.utils.config import AUDIT_DB_PATH, ensure_dirs


def init_audit_db(path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB audit store."""
    ensure_dirs()
    db_path = path or AUDIT_DB_PATH
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            decision_id VARCHAR PRIMARY KEY,
            tx_payload TEXT,
            created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            decision_id VARCHAR PRIMARY KEY,
            feat_json TEXT,
            created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            decision_id VARCHAR PRIMARY KEY,
            raw_score DOUBLE,
            calibrated_score DOUBLE,
            decision_band VARCHAR,
            model_id VARCHAR,
            calibrator_version VARCHAR,
            created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_reports (
            decision_id VARCHAR PRIMARY KEY,
            report_md TEXT,
            report_json TEXT,
            created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS replay_events (
            decision_id VARCHAR,
            replayed_at TEXT,
            outcome_match INTEGER
        )
    """)
    return con


def record_decision(
    con: duckdb.DuckDBPyConnection,
    tx_payload: dict,
    features: dict,
    prediction: dict,
    model_id: str = "rift_v1",
    calibrator_version: str = "isotonic_v1",
    report_md: Optional[str] = None,
    report_json: Optional[dict] = None,
) -> str:
    """Record a decision and return decision_id (hash)."""
    payload = {
        "tx": tx_payload,
        "features": features,
        "prediction": prediction,
        "model_id": model_id,
        "calibrator_version": calibrator_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    decision_id = canonical_hash(payload)
    now = datetime.now(timezone.utc).isoformat()

    con.execute(
        "INSERT OR REPLACE INTO transactions VALUES (?, ?, ?)",
        [decision_id, json.dumps(tx_payload, default=str), now],
    )
    con.execute(
        "INSERT OR REPLACE INTO features VALUES (?, ?, ?)",
        [decision_id, json.dumps(features, default=str), now],
    )
    con.execute(
        "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            decision_id,
            prediction.get("raw_score"),
            prediction.get("calibrated_score"),
            prediction.get("decision_band"),
            model_id,
            calibrator_version,
            now,
        ],
    )
    if report_md or report_json:
        con.execute(
            "INSERT OR REPLACE INTO audit_reports VALUES (?, ?, ?, ?)",
            [
                decision_id,
                report_md or "",
                json.dumps(report_json or {}, default=str),
                now,
            ],
        )
    return decision_id
