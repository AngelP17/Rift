from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        create table if not exists transactions (
            decision_id varchar primary key,
            created_at timestamp,
            payload_json varchar
        );
        create table if not exists features (
            decision_id varchar primary key,
            feature_json varchar
        );
        create table if not exists predictions (
            decision_id varchar primary key,
            model_run_id varchar,
            prediction_json varchar
        );
        create table if not exists audit_reports (
            decision_id varchar primary key,
            format varchar,
            report_json varchar,
            markdown varchar
        );
        create table if not exists replay_events (
            decision_id varchar,
            replayed_at timestamp,
            status varchar,
            details varchar
        );
        """
    )
    return conn


def record_decision(
    db_path: Path,
    decision_id: str,
    payload: dict[str, Any],
    feature_frame: pl.DataFrame,
    prediction: dict[str, Any],
    report: dict[str, Any],
    markdown: str,
    model_run_id: str,
) -> None:
    conn = _connect(db_path)
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
    conn.execute(
        "insert or replace into transactions values (?, ?, ?)",
        [decision_id, timestamp, json.dumps(payload, sort_keys=True)],
    )
    conn.execute(
        "insert or replace into features values (?, ?)",
        [decision_id, json.dumps(feature_frame.row(0, named=True), sort_keys=True, default=str)],
    )
    conn.execute(
        "insert or replace into predictions values (?, ?, ?)",
        [decision_id, model_run_id, json.dumps(prediction, sort_keys=True)],
    )
    conn.execute(
        "insert or replace into audit_reports values (?, ?, ?, ?)",
        [decision_id, "markdown", json.dumps(report, sort_keys=True), markdown],
    )
    conn.close()


def record_replay_event(db_path: Path, decision_id: str, status: str, details: str) -> None:
    conn = _connect(db_path)
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
    conn.execute(
        "insert into replay_events values (?, ?, ?, ?)",
        [decision_id, timestamp, status, details],
    )
    conn.close()
