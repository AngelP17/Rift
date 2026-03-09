from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient
import polars as pl

from rift.api.server import app
from rift.data.generator import generate_transactions
from rift.datasets.adapters import prepare_public_dataset
from rift.federated.simulation import list_federated_runs, train_federated_model
from rift.governance.fairness import list_fairness_audits, run_fairness_audit
from rift.models.train import train_from_frame
from rift.utils.config import get_paths


def _paths(tmp_path: Path):
    os.environ["RIFT_HOME"] = str(tmp_path / ".rift")
    return get_paths()


def test_public_dataset_adapter_prepares_ieee_cis(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = tmp_path / "ieee_sample.csv"
    pl.DataFrame(
        [
            {
                "TransactionID": 1001,
                "TransactionDT": 3600,
                "TransactionAmt": 125.5,
                "ProductCD": "W",
                "card1": 1501,
                "card4": "visa",
                "addr1": 42,
                "addr2": 12,
                "DeviceType": "desktop",
                "DeviceInfo": "Windows",
                "isFraud": 0,
            }
        ]
    ).write_csv(source)

    summary = prepare_public_dataset(source=source, adapter="ieee_cis", paths=paths, auto_etl=True)

    assert Path(summary.canonical_path).exists()
    assert summary.auto_etl_run_id is not None
    assert paths.data_path.exists()


def test_fairness_audit_generates_report(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    frame = generate_transactions(txns=800, users=80, merchants=40, fraud_rate=0.08, seed=13)
    frame.write_parquet(paths.data_path)
    train_from_frame(frame, runs_dir=paths.runs_dir, model_type="graphsage_xgb", time_split=True)

    summary = run_fairness_audit(frame=frame, paths=paths, sensitive_column="channel", threshold=0.5)

    assert Path(summary.report_path).exists()
    assert summary.demographic_parity_difference >= 0.0
    audits = list_fairness_audits(paths, limit=5)
    assert audits[0]["audit_id"] == summary.audit_id


def test_federated_training_scaffold_runs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    frame = generate_transactions(txns=600, users=60, merchants=30, fraud_rate=0.08, seed=15)
    summary = train_federated_model(
        frame=frame,
        paths=paths,
        client_column="channel",
        rounds=3,
        local_epochs=2,
        learning_rate=0.1,
        time_split=True,
    )

    assert Path(summary.artifact_path).exists()
    assert summary.client_count >= 1
    runs = list_federated_runs(paths, limit=5)
    assert runs[0]["run_id"] == summary.run_id


def test_dashboard_routes_render(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    frame = generate_transactions(txns=400, users=40, merchants=20, fraud_rate=0.08, seed=21)
    frame.write_parquet(paths.data_path)
    train_from_frame(frame, runs_dir=paths.runs_dir, model_type="xgb_tabular", time_split=False)
    run_fairness_audit(frame=frame, paths=paths, sensitive_column="channel", threshold=0.5)

    client = TestClient(app)
    dashboard_response = client.get("/dashboard")
    summary_response = client.get("/dashboard/summary")

    assert dashboard_response.status_code == 200
    assert "Rift Operations Dashboard" in dashboard_response.text
    assert summary_response.status_code == 200
    assert "fairness_audits" in summary_response.json()
