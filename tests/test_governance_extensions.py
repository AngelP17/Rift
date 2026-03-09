from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import yaml

from rift.adapters.sectors import apply_sector_profile, load_sector_profile
from rift.data.generator import generate_transactions
from rift.governance.fairness import run_fairness_audit
from rift.governance.model_cards import generate_model_card
from rift.monitoring.drift import run_drift_monitor
from rift.monitoring.nl_query import answer_natural_language_query
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.models.train import train_from_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.reengineer.simulate import simulate_legacy_migration
from rift.utils.config import get_paths
from rift.explain.report import build_audit_report, report_to_markdown


def _paths(tmp_path: Path):
    os.environ["RIFT_HOME"] = str(tmp_path / ".rift")
    os.environ["RIFT_STORAGE_BACKEND"] = "local"
    return get_paths()


def test_sector_profile_masks_and_aliases_healthcare() -> None:
    repo_root = Path("/workspace")
    frame = pl.DataFrame(
        [
            {
                "claim_id": "claim_1",
                "patient_id": "patient_1",
                "provider_id": "provider_1",
                "claim_amount": 250.0,
                "claim_timestamp": "2025-01-01T10:00:00",
                "patient_name": "Alice Smith",
            }
        ]
    )
    profiled = apply_sector_profile(frame, load_sector_profile(repo_root, "healthcare"))
    assert "tx_id" in profiled.columns
    assert profiled["patient_name"][0] == "[REDACTED]"
    assert profiled["channel"][0] == "claims_portal"


def test_green_optimization_metadata_written(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    frame = generate_transactions(txns=500, users=50, merchants=25, fraud_rate=0.08, seed=10)
    summary = train_from_frame(
        frame,
        runs_dir=paths.runs_dir,
        model_type="graphsage_xgb",
        time_split=True,
        optimize_mode="green",
    )
    metrics = json.loads(Path(summary.metadata_path).read_text(encoding="utf-8"))
    assert metrics["optimization"]["mode"] == "green"
    assert metrics["optimization"]["bytes_after"] <= metrics["optimization"]["bytes_before"]


def test_drift_monitor_and_model_card_generation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    reference = generate_transactions(txns=600, users=60, merchants=30, fraud_rate=0.05, seed=31)
    current = generate_transactions(txns=600, users=60, merchants=30, fraud_rate=0.20, seed=32).with_columns(
        (pl.col("amount") * 5).alias("amount")
    )
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    reference.write_parquet(reference_path)
    current.write_parquet(current_path)

    train_summary = train_from_frame(reference, runs_dir=paths.runs_dir, model_type="graphsage_xgb", time_split=True)
    run_fairness_audit(frame=reference, paths=paths, sensitive_column="channel", threshold=0.5)
    drift_summary = run_drift_monitor(
        paths=paths,
        reference_path=reference_path,
        current_path=current_path,
        threshold=0.05,
        trigger_retrain=False,
    )
    card_summary = generate_model_card(paths, train_summary.run_id, repo_root=Path("/workspace"))

    assert Path(drift_summary.report_path).exists()
    assert Path(card_summary.model_card_path).exists()
    assert "Model Card" in Path(card_summary.model_card_path).read_text(encoding="utf-8")


def test_natural_language_query_returns_results(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    frame = generate_transactions(txns=500, users=50, merchants=25, fraud_rate=0.08, seed=33)
    summary = train_from_frame(frame, runs_dir=paths.runs_dir, model_type="xgb_tabular", time_split=True)
    artifact = load_run(paths.runs_dir, summary.run_id)
    payload = {
        "tx_id": "demo_query_tx",
        "user_id": "user_00001",
        "merchant_id": "merchant_00001",
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
    record_decision(
        db_path=paths.audit_db,
        decision_id=decision_id,
        payload=payload,
        feature_frame=feature_frame,
        prediction=prediction,
        report=report,
        markdown=report_to_markdown(report),
        model_run_id=summary.run_id,
    )

    result = answer_natural_language_query(paths, "show recent flagged transactions")
    assert Path(result.result_path).exists()
    assert result.sql


def test_legacy_migration_simulation_and_configs_exist(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    legacy_source = tmp_path / "legacy.csv"
    pl.DataFrame(
        [
            {
                "transaction_id": "legacy_1",
                "beneficiary_id": "case_1",
                "vendor_id": "vendor_1",
                "amount": 100.0,
                "event_time": "2025-01-01T10:00:00",
            }
        ]
    ).write_csv(legacy_source)
    summary = simulate_legacy_migration(
        paths=paths,
        source=legacy_source,
        output_path=tmp_path / "legacy.parquet",
        sector="fintech",
    )
    assert Path(summary.output_path).exists()
    assert summary.rows_loaded == 1

    workflow = yaml.safe_load(Path("/workspace/.github/workflows/validate.yml").read_text(encoding="utf-8"))
    hub = yaml.safe_load(Path("/workspace/docker/jupyterhub.yml").read_text(encoding="utf-8"))
    assert "jobs" in workflow
    assert "services" in hub
