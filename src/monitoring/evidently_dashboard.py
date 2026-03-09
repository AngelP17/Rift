"""Evidently AI monitoring for data drift, target drift, and model quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


def generate_drift_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate an Evidently drift report comparing reference and current data."""
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
        from evidently.report import Report

        ref_df = pd.read_parquet(reference_path)
        cur_df = pd.read_parquet(current_path)

        numeric_cols = ref_df.select_dtypes(include=["number"]).columns.tolist()
        cols_to_use = [c for c in numeric_cols if c in cur_df.columns]
        ref_df = ref_df[cols_to_use]
        cur_df = cur_df[cols_to_use]

        presets = [DataDriftPreset(), DataQualityPreset()]
        if "is_fraud" in ref_df.columns:
            presets.append(TargetDriftPreset())

        report = Report(metrics=presets)
        report.run(reference_data=ref_df, current_data=cur_df)

        output_path = output_path or (cfg.data_dir / "reports" / "evidently_drift.html")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))

        result_json = report.as_dict()
        n_drifted = 0
        for metric in result_json.get("metrics", []):
            r = metric.get("result", {})
            if r.get("dataset_drift") is True:
                n_drifted += 1

        summary = {
            "report_path": str(output_path),
            "n_metrics": len(result_json.get("metrics", [])),
            "dataset_drift_detected": n_drifted > 0,
            "n_drifted_features": n_drifted,
        }

        log.info("evidently_drift_report", **summary)
        return summary

    except ImportError:
        log.warning("evidently_not_installed", msg="pip install evidently")
        return {"error": "evidently not installed", "install": "pip install evidently"}


def generate_model_quality_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a model quality report with Evidently."""
    try:
        from evidently.metric_preset import ClassificationPreset
        from evidently.report import Report

        ref_df = pd.read_parquet(reference_path)
        cur_df = pd.read_parquet(current_path)

        report = Report(metrics=[ClassificationPreset()])

        numeric_cols = ref_df.select_dtypes(include=["number"]).columns.tolist()
        cols_to_use = [c for c in numeric_cols if c in cur_df.columns]

        report.run(reference_data=ref_df[cols_to_use], current_data=cur_df[cols_to_use])

        output_path = output_path or (cfg.data_dir / "reports" / "evidently_quality.html")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))

        return {"report_path": str(output_path)}

    except ImportError:
        return {"error": "evidently not installed"}
    except Exception as e:
        return {"error": str(e)}


def launch_streamlit_dashboard() -> str:
    """Return the command to launch the Streamlit monitoring dashboard."""
    dashboard_path = Path(__file__).parent / "streamlit_app.py"
    return f"streamlit run {dashboard_path} --server.port 8501"
