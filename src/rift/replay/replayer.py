from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from rift.replay.recorder import record_replay_event


def fetch_decision(db_path: Path, decision_id: str) -> dict[str, Any]:
    conn = duckdb.connect(str(db_path), read_only=True)
    tx = conn.execute(
        "select payload_json from transactions where decision_id = ?",
        [decision_id],
    ).fetchone()
    pred = conn.execute(
        "select model_run_id, prediction_json from predictions where decision_id = ?",
        [decision_id],
    ).fetchone()
    report = conn.execute(
        "select report_json, markdown from audit_reports where decision_id = ?",
        [decision_id],
    ).fetchone()
    conn.close()
    if tx is None or pred is None or report is None:
        raise KeyError(f"decision_id not found: {decision_id}")
    return {
        "payload": json.loads(tx[0]),
        "model_run_id": pred[0],
        "prediction": json.loads(pred[1]),
        "report": json.loads(report[0]),
        "markdown": report[1],
    }


def replay_decision(db_path: Path, decision_id: str) -> dict[str, Any]:
    decision = fetch_decision(db_path, decision_id)
    record_replay_event(db_path, decision_id, "success", "stored decision fetched")
    return decision
