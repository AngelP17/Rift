from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from rift.data.generator import generate_transactions
from rift.explain.report import build_audit_report, report_to_markdown
from rift.features.engine import build_features
from rift.graph.builder import build_transaction_graph
from rift.models.calibrate import ProbabilityCalibrator
from rift.models.conformal import ConformalClassifier
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.models.train import train_from_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
from rift.utils.config import get_paths


def _train_fixture(tmp_path: Path):
    os.environ["RIFT_HOME"] = str(tmp_path / ".rift")
    paths = get_paths()
    frame = generate_transactions(txns=800, users=80, merchants=40, fraud_rate=0.08, seed=9)
    frame.write_parquet(paths.data_path)
    summary = train_from_frame(frame, runs_dir=paths.runs_dir, model_type="graphsage_xgb", time_split=True)
    artifact = load_run(paths.runs_dir, summary.run_id)
    return paths, frame, summary, artifact


def test_generator_has_required_columns() -> None:
    frame = generate_transactions(txns=200, users=20, merchants=10, fraud_rate=0.1, seed=11)
    assert frame.height == 200
    for column in [
        "tx_id",
        "user_id",
        "merchant_id",
        "device_id",
        "account_id",
        "amount",
        "timestamp",
        "channel",
        "mcc",
        "is_fraud",
    ]:
        assert column in frame.columns


def test_feature_engine_outputs_temporal_columns() -> None:
    frame = generate_transactions(txns=120, users=10, merchants=8, fraud_rate=0.05, seed=2)
    feature_frame = build_features(frame)
    assert "user_txn_count_1h" in feature_frame.columns
    assert "merchant_fraud_prevalence" in feature_frame.columns


def test_graph_builder_creates_edges() -> None:
    frame = generate_transactions(txns=120, users=10, merchants=8, fraud_rate=0.05, seed=5)
    graph = build_transaction_graph(frame)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0


def test_training_pipeline_writes_artifacts(tmp_path: Path) -> None:
    paths, _, summary, _ = _train_fixture(tmp_path)
    assert Path(summary.artifact_path).exists()
    assert summary.metrics["pr_auc"] >= 0.0
    assert (paths.runs_dir / "current_run.json").exists()


def test_calibration_and_conformal_outputs_valid_range() -> None:
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    calibrated = calibrator.predict([0.15, 0.85])
    conformal = ConformalClassifier(alpha=0.1)
    conformal.fit(calibrated, [0, 1])
    decisions = conformal.triage(calibrated)
    assert all(0.0 <= value <= 1.0 for value in calibrated)
    assert decisions[0][0] in {"high_confidence_legit", "high_confidence_fraud", "review_needed"}


def test_predict_record_and_replay(tmp_path: Path) -> None:
    paths, _, summary, artifact = _train_fixture(tmp_path)
    payload = {
        "tx_id": "demo_predict_tx",
        "user_id": "user_00001",
        "merchant_id": "merchant_00002",
        "device_id": "device_00001",
        "account_id": "acct_00001",
        "amount": 999.0,
        "currency": "USD",
        "timestamp": "2025-03-03T12:00:00",
        "lat": 40.0,
        "lon": -74.0,
        "channel": "web",
        "mcc": "electronics",
    }
    prediction, feature_frame = score_frame(payload_to_frame(payload), artifact)
    decision_id = decision_hash({"payload": payload, "prediction": prediction, "run_id": summary.run_id})
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
        model_run_id=summary.run_id,
    )
    replayed = replay_decision(paths.audit_db, decision_id)
    assert replayed["prediction"]["decision"] == prediction["decision"]
    assert decision_id in replayed["markdown"]


def test_report_renderer_contains_core_sections(tmp_path: Path) -> None:
    _, _, summary, artifact = _train_fixture(tmp_path)
    payload = {
        "tx_id": "demo_report_tx",
        "user_id": "user_00003",
        "merchant_id": "merchant_00008",
        "device_id": "device_00003",
        "account_id": "acct_00003",
        "amount": 430.0,
        "currency": "USD",
        "timestamp": "2025-03-03T12:00:00",
        "lat": 38.0,
        "lon": -76.0,
        "channel": "mobile",
        "mcc": "travel",
    }
    prediction, feature_frame = score_frame(payload_to_frame(payload), artifact)
    decision_id = decision_hash({"payload": payload, "prediction": prediction, "run_id": summary.run_id})
    report = build_audit_report(decision_id, feature_frame, artifact, prediction)
    markdown = report_to_markdown(report)
    assert "Top drivers" in markdown
    assert "Counterfactual summary" in markdown
