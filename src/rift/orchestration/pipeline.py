from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rift.data.generator import generate_transactions
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.models.train import train_from_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.utils.config import RiftPaths
from rift.utils.io import read_json, write_json


@dataclass(frozen=True)
class PipelineRunSummary:
    generated_rows: int
    train_run_id: str
    decision_id: str
    audit_decision: str
    report_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_end_to_end_pipeline(
    paths: RiftPaths,
    txns: int,
    users: int,
    merchants: int,
    fraud_rate: float,
    model_type: str,
    sample_tx_path: Path,
    optimize_mode: str = "standard",
) -> PipelineRunSummary:
    frame = generate_transactions(txns=txns, users=users, merchants=merchants, fraud_rate=fraud_rate, seed=7)
    frame.write_parquet(paths.data_path)
    train_summary = train_from_frame(
        frame,
        runs_dir=paths.runs_dir,
        model_type=model_type,
        time_split=True,
        optimize_mode=optimize_mode,
    )

    artifact = load_run(paths.runs_dir, train_summary.run_id)
    payload = read_json(sample_tx_path)
    payload_frame = payload_to_frame(payload)
    prediction, feature_frame = score_frame(payload_frame, artifact)
    explanation, _ = build_explanation(feature_frame, artifact, prediction)
    prediction["explanation"] = explanation
    decision_id = decision_hash({"payload": payload, "model_run_id": train_summary.run_id, "prediction": prediction})
    report = build_audit_report(decision_id, feature_frame, artifact, prediction)
    markdown = report_to_markdown(report)
    record_decision(
        db_path=paths.audit_db,
        decision_id=decision_id,
        payload=payload,
        feature_frame=feature_frame,
        prediction=prediction,
        report=report,
        markdown=markdown,
        model_run_id=train_summary.run_id,
    )
    report_path = paths.lakehouse_dir / f"{decision_id}_audit_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    summary = PipelineRunSummary(
        generated_rows=frame.height,
        train_run_id=train_summary.run_id,
        decision_id=decision_id,
        audit_decision=prediction["decision"],
        report_path=str(report_path),
    )
    write_json(paths.lakehouse_dir / "latest_pipeline_run.json", summary.to_dict())
    return summary
