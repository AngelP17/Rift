"""Dashboard views: data loading, template rendering, and drill-down builders."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from jinja2 import Environment, FileSystemLoader

from rift.dashboard.kpis import (
    EMPTY_STATES,
    QUICK_ACTIONS,
    build_kpi_cards,
)
from rift.datasets.adapters import list_prepared_datasets
from rift.etl.pipeline import list_etl_runs
from rift.federated.simulation import list_federated_runs
from rift.governance.fairness import list_fairness_audits
from rift.monitoring.drift import list_drift_reports
from rift.storage.backends import get_storage_backend
from rift.utils.config import RiftPaths
from rift.utils.io import read_json

VERSION = "1.0.0"

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=True,
)


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _recent_audits(paths: RiftPaths, limit: int = 10) -> list[dict[str, Any]]:
    if not paths.audit_db.exists():
        return []
    conn = duckdb.connect(str(paths.audit_db), read_only=True)
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'predictions'"
    ).fetchone()[0]
    if not exists:
        conn.close()
        return []
    rows = conn.execute(
        """
        select p.decision_id, p.model_run_id, p.prediction_json, ar.markdown
        from predictions p
        left join audit_reports ar on ar.decision_id = p.decision_id
        order by p.decision_id desc
        limit ?
        """,
        [limit],
    ).fetchall()
    conn.close()
    output: list[dict[str, Any]] = []
    for decision_id, model_run_id, prediction_json, markdown in rows:
        prediction = json.loads(prediction_json)
        output.append(
            {
                "decision_id": decision_id,
                "model_run_id": model_run_id,
                "decision": prediction.get("decision"),
                "calibrated_probability": prediction.get("calibrated_probability"),
                "confidence": prediction.get("confidence"),
                "markdown": markdown,
            }
        )
    return output


def dashboard_snapshot(paths: RiftPaths) -> dict[str, Any]:
    current_run = _safe_read_json(paths.runs_dir / "current_run.json")
    current_metrics = None
    if current_run:
        current_metrics = _safe_read_json(paths.runs_dir / current_run["run_id"] / "metrics.json")
    etl_runs = list_etl_runs(paths.warehouse_db, limit=10)
    fairness_runs = list_fairness_audits(paths, limit=10)
    drift_reports = list_drift_reports(paths, limit=10)
    federated_runs = list_federated_runs(paths, limit=10)
    prepared_datasets = list_prepared_datasets(paths, limit=10)
    recent_audits = _recent_audits(paths, limit=10)
    storage_status = get_storage_backend(paths).status().to_dict()
    return {
        "current_model": current_run,
        "current_metrics": current_metrics,
        "etl_runs": etl_runs,
        "fairness_audits": fairness_runs,
        "drift_reports": drift_reports,
        "federated_runs": federated_runs,
        "prepared_datasets": prepared_datasets,
        "recent_audits": recent_audits,
        "storage_status": storage_status,
        "kpis": {
            "etl_runs": len(etl_runs),
            "fairness_audits": len(fairness_runs),
            "drift_reports": len(drift_reports),
            "federated_runs": len(federated_runs),
            "recent_audits": len(recent_audits),
        },
    }


def _base_context(paths: RiftPaths, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    if snapshot is None:
        snapshot = dashboard_snapshot(paths)
    current_run = snapshot.get("current_model")
    return {
        "version": VERSION,
        "commit_sha": _git_short_sha(),
        "run_id": current_run["run_id"] if current_run else "none",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_dashboard_html(paths: RiftPaths) -> str:
    snapshot = dashboard_snapshot(paths)
    kpi_cards = build_kpi_cards(snapshot["kpis"], snapshot["current_metrics"])
    prepared_rows = [item.get("summary", {}) for item in snapshot["prepared_datasets"]]

    ctx = {
        **_base_context(paths, snapshot),
        "kpi_cards": [c.to_dict() for c in kpi_cards],
        "quick_actions": [asdict(a) for a in QUICK_ACTIONS],
        "etl_runs": snapshot["etl_runs"],
        "fairness_audits": snapshot["fairness_audits"],
        "drift_reports": snapshot["drift_reports"],
        "federated_runs": snapshot["federated_runs"],
        "prepared_datasets": prepared_rows,
        "recent_audits": snapshot["recent_audits"],
        "storage_status": snapshot["storage_status"],
        "empty_states": {k: asdict(v) for k, v in EMPTY_STATES.items()},
    }
    return _env.get_template("index.html").render(**ctx)


def build_detail_html(
    paths: RiftPaths,
    page_title: str,
    page_description: str,
    sections: list[dict[str, Any]],
) -> str:
    snapshot = dashboard_snapshot(paths)
    ctx = {
        **_base_context(paths, snapshot),
        "page_title": page_title,
        "page_description": page_description,
        "sections": sections,
    }
    return _env.get_template("detail.html").render(**ctx)


def build_etl_detail(paths: RiftPaths) -> str:
    etl_runs = list_etl_runs(paths.warehouse_db, limit=50)
    return build_detail_html(
        paths, "ETL Pipeline Runs",
        "Auditable bronze/silver/gold ETL pipeline executions with data quality tracking.",
        [{"title": "All ETL Runs",
          "columns": ["run_id", "source_system", "dataset_name", "rows_valid", "rows_invalid", "duplicates_removed"],
          "rows": etl_runs,
          "empty_state": asdict(EMPTY_STATES["etl"]) if not etl_runs else None}],
    )


def build_governance_detail(paths: RiftPaths) -> str:
    fairness = list_fairness_audits(paths, limit=50)
    drift = list_drift_reports(paths, limit=50)
    return build_detail_html(
        paths, "Governance & Compliance",
        "Fairness audits, drift monitoring reports, and model card generation status.",
        [
            {"title": "Fairness Audits",
             "columns": ["audit_id", "sensitive_column", "demographic_parity_difference", "disparate_impact_ratio", "report_path"],
             "rows": fairness,
             "empty_state": asdict(EMPTY_STATES["fairness"]) if not fairness else None},
            {"title": "Drift Reports",
             "columns": ["report_id", "drift_score", "is_drift", "retrain_triggered", "report_path"],
             "rows": drift,
             "empty_state": asdict(EMPTY_STATES["drift"]) if not drift else None},
        ],
    )


def build_audits_detail(paths: RiftPaths) -> str:
    audits = _recent_audits(paths, limit=50)
    return build_detail_html(
        paths, "Audit Decision Records",
        "SHA-256 hashed, replayable decision records stored in DuckDB.",
        [{"title": "All Recorded Decisions",
          "columns": ["decision_id", "model_run_id", "decision", "calibrated_probability", "confidence"],
          "rows": audits,
          "empty_state": asdict(EMPTY_STATES["audits"]) if not audits else None}],
    )


def build_models_detail(paths: RiftPaths) -> str:
    current_run = _safe_read_json(paths.runs_dir / "current_run.json")
    current_metrics = None
    if current_run:
        current_metrics = _safe_read_json(paths.runs_dir / current_run["run_id"] / "metrics.json")

    model_rows = []
    if current_run and current_metrics:
        m = current_metrics.get("metrics", {})
        model_rows.append({
            "run_id": current_run["run_id"],
            "model_type": current_metrics.get("model_type", ""),
            "pr_auc": f"{m.get('pr_auc', 0):.4f}",
            "ece": f"{m.get('ece', 0):.4f}",
            "brier_score": f"{m.get('brier_score', 0):.4f}",
            "optimization": current_metrics.get("optimization", {}).get("mode", "default"),
        })

    federated = list_federated_runs(paths, limit=20)
    return build_detail_html(
        paths, "Model Runs & Performance",
        "Training runs, model metrics, optimization modes, and federated scaffolds.",
        [
            {"title": "Current Model",
             "columns": ["run_id", "model_type", "pr_auc", "ece", "brier_score", "optimization"],
             "rows": model_rows,
             "empty_state": asdict(EMPTY_STATES["models"]) if not model_rows else None},
            {"title": "Federated Training Runs",
             "columns": ["run_id", "client_column", "client_count", "rounds"],
             "rows": federated,
             "empty_state": asdict(EMPTY_STATES["federated"]) if not federated else None},
        ],
    )


def get_static_dir() -> Path:
    return _STATIC_DIR
