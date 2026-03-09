from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from rift.governance.fairness import list_fairness_audits
from rift.monitoring.drift import list_drift_reports
from rift.utils.config import RiftPaths, get_repo_root
from rift.utils.io import read_json, read_pickle


@dataclass(frozen=True)
class ModelCardSummary:
    run_id: str
    model_card_path: str
    governance_summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _template_env(root: Path) -> Environment:
    return Environment(loader=FileSystemLoader(str(root / "docs" / "templates")), autoescape=False)


def _latest_by_run(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("run_id") == run_id:
            return row
    return None


def generate_model_card(paths: RiftPaths, run_id: str, repo_root: Path | None = None) -> ModelCardSummary:
    metadata = read_json(paths.runs_dir / run_id / "metrics.json")
    artifact = read_pickle(paths.runs_dir / run_id / "artifact.pkl")
    env = _template_env(repo_root or get_repo_root())
    fairness = _latest_by_run(list_fairness_audits(paths, limit=50), run_id)
    drift = _latest_by_run(list_drift_reports(paths, limit=50), run_id)
    optimization = artifact.get("optimization", {"mode": "standard", "bytes_before": 0, "bytes_after": 0, "reduction_ratio": 0.0})
    context = {
        "model_name": "Rift Fraud Model",
        "run_id": run_id,
        "model_type": metadata.get("model_type"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trained_at": artifact.get("trained_at"),
        "mlflow_run_id": metadata.get("mlflow_run_id"),
        "data_path": str(paths.data_path),
        "feature_count": len(metadata.get("feature_columns", [])),
        "time_split": metadata.get("time_split"),
        "calibration_method": metadata.get("calibration_method"),
        "sector_profile": artifact.get("sector_profile", "fintech"),
        "metrics": metadata.get("metrics", {}),
        "latest_fairness": fairness,
        "latest_drift": drift,
        "optimization": optimization,
    }
    model_card = env.get_template("model_card.md.j2").render(**context)
    governance_summary = env.get_template("governance_summary.md.j2").render(
        run_id=run_id,
        model_type=metadata.get("model_type"),
        generated_at=context["generated_at"],
        latest_fairness=fairness,
        latest_drift=drift,
    )
    model_card_path = paths.model_cards_dir / f"{run_id}.md"
    governance_path = paths.model_cards_dir / f"{run_id}_governance.md"
    model_card_path.write_text(model_card, encoding="utf-8")
    governance_path.write_text(governance_summary, encoding="utf-8")
    return ModelCardSummary(
        run_id=run_id,
        model_card_path=str(model_card_path),
        governance_summary_path=str(governance_path),
    )
