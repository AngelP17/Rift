from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import polars as pl
import yaml

from rift.data.generator import generate_transactions
from rift.lakehouse.sql import build_default_views, query_lakehouse
from rift.orchestration.pipeline import run_end_to_end_pipeline
from rift.storage.backends import get_storage_backend
from rift.utils.config import get_paths


def _paths(tmp_path: Path):
    os.environ["RIFT_HOME"] = str(tmp_path / ".rift")
    os.environ["RIFT_STORAGE_BACKEND"] = "local"
    return get_paths()


def test_local_storage_backend_saves_and_loads_parquet(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    backend = get_storage_backend(paths)
    frame = pl.DataFrame([{"tx_id": "tx_1", "amount": 12.5}])
    target = backend.save_parquet(frame, "objects/test_frame.parquet")

    loaded = backend.load_parquet("objects/test_frame.parquet")

    assert Path(target).exists()
    assert loaded.height == 1
    assert backend.status().backend == "local"


def test_lakehouse_query_reads_default_views(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    frame = generate_transactions(txns=120, users=12, merchants=6, fraud_rate=0.05, seed=4)
    frame.write_parquet(paths.data_path)
    feature_frame = frame.head(20)
    feature_frame.write_parquet(paths.feature_store_path)

    build_default_views(paths)
    result = query_lakehouse(paths, "select count(*) as row_count from transactions")

    assert result.rows == 1
    assert result.preview[0]["row_count"] == 120


def test_end_to_end_pipeline_runner_creates_report(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    sample_tx = tmp_path / "sample_tx.json"
    sample_tx.write_text(
        """
{
  "tx_id": "demo_tx_local_1",
  "user_id": "user_00001",
  "merchant_id": "merchant_00002",
  "device_id": "device_00001",
  "account_id": "acct_00001",
  "amount": 450.5,
  "currency": "USD",
  "timestamp": "2025-03-01T10:30:00",
  "lat": 40.0,
  "lon": -74.0,
  "channel": "web",
  "mcc": "electronics"
}
        """.strip(),
        encoding="utf-8",
    )

    summary = run_end_to_end_pipeline(
        paths=paths,
        txns=600,
        users=60,
        merchants=24,
        fraud_rate=0.08,
        model_type="xgb_tabular",
        sample_tx_path=sample_tx,
    )

    assert Path(summary.report_path).exists()
    assert summary.generated_rows == 600


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_airflow_dag_module_imports_safely() -> None:
    dag_path = _repo_root() / "dags/rift_pipeline.py"
    spec = importlib.util.spec_from_file_location("rift_pipeline", dag_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "dag")


def test_docker_compose_file_is_valid_yaml() -> None:
    compose_path = _repo_root() / "docker-compose.yml"
    payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert "services" in payload
    assert "minio" in payload["services"]
    assert "airflow-webserver" in payload["services"]
