from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from rift.features.engine import build_features
from rift.graph.builder import build_transaction_graph
from rift.models.infer import load_run
from rift.models.train import _select
from rift.utils.config import RiftPaths
from rift.utils.io import write_json


@dataclass(frozen=True)
class FairnessAuditSummary:
    audit_id: str
    run_id: str
    data_path: str
    sensitive_column: str
    threshold: float
    report_path: str
    demographic_parity_difference: float
    disparate_impact_ratio: float
    equal_opportunity_difference: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _predict_probabilities(frame: pl.DataFrame, artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    feature_frame = build_features(frame, categorical_mappings=artifact.get("categorical_mappings"))
    matrix = _select(feature_frame, artifact["feature_columns"])
    if artifact["model_type"] == "xgb_tabular":
        raw = artifact["model"].predict_proba(matrix)
    else:
        raw = artifact["model"].predict_proba(matrix, build_transaction_graph(feature_frame))
    calibrated = artifact["calibrator"].predict(raw)
    return raw, calibrated


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _group_metrics(groups: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    predictions = probabilities >= threshold
    metrics: list[dict[str, Any]] = []
    for group in sorted({str(value) for value in groups}):
        mask = np.asarray([str(value) == group for value in groups], dtype=bool)
        group_labels = labels[mask]
        group_probs = probabilities[mask]
        group_pred = predictions[mask]
        positives = int(group_pred.sum())
        negatives = int((~group_pred).sum())
        actual_positive = int(group_labels.sum())
        actual_negative = int((1 - group_labels).sum())
        true_positive = int(((group_pred == 1) & (group_labels == 1)).sum())
        false_positive = int(((group_pred == 1) & (group_labels == 0)).sum())
        metrics.append(
            {
                "group": group,
                "count": int(mask.sum()),
                "avg_score": float(group_probs.mean()) if group_probs.size else 0.0,
                "selection_rate": float(group_pred.mean()) if group_pred.size else 0.0,
                "fraud_rate": float(group_labels.mean()) if group_labels.size else 0.0,
                "true_positive_rate": _safe_rate(true_positive, actual_positive),
                "false_positive_rate": _safe_rate(false_positive, actual_negative),
                "positive_predictions": positives,
                "negative_predictions": negatives,
            }
        )
    return metrics


def _summary_from_groups(group_metrics: list[dict[str, Any]]) -> tuple[float, float, float | None]:
    if not group_metrics:
        return 0.0, 1.0, None
    selection_rates = [float(metric["selection_rate"]) for metric in group_metrics]
    demographic_parity_difference = max(selection_rates) - min(selection_rates)
    max_rate = max(selection_rates)
    min_rate = min(selection_rates)
    disparate_impact_ratio = float(min_rate / max_rate) if max_rate > 0 else 1.0
    tprs = [metric["true_positive_rate"] for metric in group_metrics if metric["true_positive_rate"] is not None]
    equal_opportunity_difference = float(max(tprs) - min(tprs)) if tprs else None
    return demographic_parity_difference, disparate_impact_ratio, equal_opportunity_difference


def _store_report(paths: RiftPaths, summary: FairnessAuditSummary, report: dict[str, Any]) -> None:
    report_path = Path(summary.report_path)
    write_json(report_path, report)
    conn = duckdb.connect(str(paths.governance_db))
    conn.execute(
        """
        create table if not exists fairness_audits (
            audit_id varchar primary key,
            run_id varchar,
            data_path varchar,
            sensitive_column varchar,
            threshold double,
            demographic_parity_difference double,
            disparate_impact_ratio double,
            equal_opportunity_difference double,
            report_path varchar,
            created_at timestamp
        )
        """
    )
    conn.execute(
        """
        insert or replace into fairness_audits values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            summary.audit_id,
            summary.run_id,
            summary.data_path,
            summary.sensitive_column,
            summary.threshold,
            summary.demographic_parity_difference,
            summary.disparate_impact_ratio,
            summary.equal_opportunity_difference,
            summary.report_path,
            datetime.now(timezone.utc).replace(tzinfo=None),
        ],
    )
    conn.close()


def run_fairness_audit(
    frame: pl.DataFrame,
    paths: RiftPaths,
    sensitive_column: str,
    run_id: str | None = None,
    threshold: float = 0.5,
    data_path: str | None = None,
) -> FairnessAuditSummary:
    if sensitive_column not in frame.columns:
        raise ValueError(f"sensitive column '{sensitive_column}' not found in frame")

    artifact = load_run(paths.runs_dir, run_id=run_id)
    labels = frame["is_fraud"].to_numpy().astype(int) if "is_fraud" in frame.columns else np.zeros(frame.height, dtype=int)
    _, calibrated = _predict_probabilities(frame, artifact)
    group_metrics = _group_metrics(frame[sensitive_column].to_numpy(), labels, calibrated, threshold)
    demographic_parity_difference, disparate_impact_ratio, equal_opportunity_difference = _summary_from_groups(group_metrics)

    audit_id = f"fairness_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    report_path = paths.fairness_dir / f"{audit_id}.json"
    summary = FairnessAuditSummary(
        audit_id=audit_id,
        run_id=artifact["_run_id"],
        data_path=data_path or str(paths.data_path),
        sensitive_column=sensitive_column,
        threshold=threshold,
        report_path=str(report_path),
        demographic_parity_difference=demographic_parity_difference,
        disparate_impact_ratio=disparate_impact_ratio,
        equal_opportunity_difference=equal_opportunity_difference,
    )
    report = {
        "summary": summary.to_dict(),
        "group_metrics": group_metrics,
        "overall": {
            "row_count": frame.height,
            "mean_calibrated_score": float(np.mean(calibrated)) if calibrated.size else 0.0,
            "positive_prediction_rate": float(np.mean(calibrated >= threshold)) if calibrated.size else 0.0,
        },
    }
    _store_report(paths, summary, report)
    return summary


def list_fairness_audits(paths: RiftPaths, limit: int = 10) -> list[dict[str, Any]]:
    if not paths.governance_db.exists():
        return []
    conn = duckdb.connect(str(paths.governance_db), read_only=True)
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'fairness_audits'"
    ).fetchone()[0]
    if not exists:
        conn.close()
        return []
    rows = conn.execute(
        """
        select audit_id, run_id, data_path, sensitive_column, threshold,
               demographic_parity_difference, disparate_impact_ratio,
               equal_opportunity_difference, report_path, created_at
        from fairness_audits
        order by created_at desc
        limit ?
        """,
        [limit],
    ).fetchall()
    conn.close()
    keys = [
        "audit_id",
        "run_id",
        "data_path",
        "sensitive_column",
        "threshold",
        "demographic_parity_difference",
        "disparate_impact_ratio",
        "equal_opportunity_difference",
        "report_path",
        "created_at",
    ]
    return [dict(zip(keys, row, strict=False)) for row in rows]
