from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from rift.features.engine import build_features, feature_columns
from rift.models.train import train_from_frame
from rift.utils.config import RiftPaths
from rift.utils.io import read_json, write_json


@dataclass(frozen=True)
class DriftReportSummary:
    report_id: str
    run_id: str | None
    reference_path: str
    current_path: str
    drift_score: float
    is_drift: bool
    threshold: float
    retrain_triggered: bool
    retrain_run_id: str | None
    report_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if suffix == ".json":
        return pl.read_json(path)
    raise ValueError(f"unsupported data format: {path.suffix}")


def _feature_matrix(frame: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    feat = build_features(frame)
    cols = feature_columns(feat)
    return feat.select(cols).to_numpy().astype(float), cols


def _alibi_drift_score(reference: np.ndarray, current: np.ndarray) -> tuple[float, bool] | None:
    try:
        from alibi_detect.cd import TabularDrift
    except Exception:
        return None
    detector = TabularDrift(reference, p_val=0.05)
    result = detector.predict(current)
    p_values = np.asarray(result["data"].get("p_val", []), dtype=float)
    if p_values.size == 0:
        return None
    score = float(np.mean(1.0 - p_values))
    return score, bool(result["data"].get("is_drift", 0))


def _fallback_drift_score(reference: np.ndarray, current: np.ndarray) -> tuple[float, bool]:
    ref_mean = np.mean(reference, axis=0)
    cur_mean = np.mean(current, axis=0)
    ref_std = np.std(reference, axis=0)
    ref_std[ref_std == 0.0] = 1.0
    z = np.abs(cur_mean - ref_mean) / ref_std
    score = float(np.mean(np.clip(z / 3.0, 0.0, 1.0)))
    return score, bool(score >= 0.2)


def _store_report(paths: RiftPaths, summary: DriftReportSummary, payload: dict[str, Any]) -> None:
    report_path = Path(summary.report_path)
    write_json(report_path, payload)
    conn = duckdb.connect(str(paths.governance_db))
    conn.execute(
        """
        create table if not exists drift_reports (
            report_id varchar primary key,
            run_id varchar,
            reference_path varchar,
            current_path varchar,
            drift_score double,
            is_drift boolean,
            threshold double,
            retrain_triggered boolean,
            retrain_run_id varchar,
            report_path varchar,
            created_at timestamp
        )
        """
    )
    conn.execute(
        """
        insert or replace into drift_reports values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            summary.report_id,
            summary.run_id,
            summary.reference_path,
            summary.current_path,
            summary.drift_score,
            summary.is_drift,
            summary.threshold,
            summary.retrain_triggered,
            summary.retrain_run_id,
            summary.report_path,
            datetime.now(timezone.utc).replace(tzinfo=None),
        ],
    )
    conn.close()


def run_drift_monitor(
    paths: RiftPaths,
    reference_path: Path,
    current_path: Path,
    threshold: float = 0.2,
    trigger_retrain: bool = False,
    model_type: str = "graphsage_xgb",
) -> DriftReportSummary:
    reference_frame = _read_frame(reference_path)
    current_frame = _read_frame(current_path)
    reference_matrix, columns = _feature_matrix(reference_frame)
    current_matrix, _ = _feature_matrix(current_frame)
    alibi_result = _alibi_drift_score(reference_matrix, current_matrix)
    if alibi_result is None:
        drift_score, detected = _fallback_drift_score(reference_matrix, current_matrix)
        detector = "fallback_zscore"
    else:
        drift_score, detected = alibi_result
        detector = "alibi_detect"
    is_drift = bool(detected or drift_score >= threshold)

    retrain_run_id = None
    if is_drift and trigger_retrain:
        train_summary = train_from_frame(current_frame, runs_dir=paths.runs_dir, model_type=model_type, time_split=True)
        retrain_run_id = train_summary.run_id
    current_run_id = None
    current_run_file = paths.runs_dir / "current_run.json"
    if current_run_file.exists():
        current_run_id = read_json(current_run_file).get("run_id")

    report_id = f"drift_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    report_path = paths.drift_dir / f"{report_id}.json"
    summary = DriftReportSummary(
        report_id=report_id,
        run_id=retrain_run_id or current_run_id,
        reference_path=str(reference_path),
        current_path=str(current_path),
        drift_score=drift_score,
        is_drift=is_drift,
        threshold=threshold,
        retrain_triggered=bool(retrain_run_id),
        retrain_run_id=retrain_run_id,
        report_path=str(report_path),
    )
    payload = {
        "summary": summary.to_dict(),
        "detector": detector,
        "feature_columns": columns,
        "reference_rows": reference_frame.height,
        "current_rows": current_frame.height,
    }
    _store_report(paths, summary, payload)
    return summary


def list_drift_reports(paths: RiftPaths, limit: int = 10) -> list[dict[str, Any]]:
    if not paths.governance_db.exists():
        return []
    conn = duckdb.connect(str(paths.governance_db), read_only=True)
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'drift_reports'"
    ).fetchone()[0]
    if not exists:
        conn.close()
        return []
    rows = conn.execute(
        """
        select report_id, run_id, reference_path, current_path, drift_score, is_drift,
               threshold, retrain_triggered, retrain_run_id, report_path, created_at
        from drift_reports
        order by created_at desc
        limit ?
        """,
        [limit],
    ).fetchall()
    conn.close()
    keys = [
        "report_id",
        "run_id",
        "reference_path",
        "current_path",
        "drift_score",
        "is_drift",
        "threshold",
        "retrain_triggered",
        "retrain_run_id",
        "report_path",
        "created_at",
    ]
    return [dict(zip(keys, row, strict=False)) for row in rows]
