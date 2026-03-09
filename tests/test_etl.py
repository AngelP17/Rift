from __future__ import annotations

import os
from pathlib import Path

import duckdb
import polars as pl

from rift.etl.pipeline import list_etl_runs, run_etl_pipeline
from rift.utils.config import get_paths


def _etl_paths(tmp_path: Path):
    os.environ["RIFT_HOME"] = str(tmp_path / ".rift")
    return get_paths()


def test_etl_pipeline_normalizes_and_redacts(tmp_path: Path) -> None:
    paths = _etl_paths(tmp_path)
    source = tmp_path / "government_source.csv"
    pl.DataFrame(
        [
            {
                "transaction_id": "gov_1",
                "beneficiary_id": "case_001",
                "vendor_id": "vendor_100",
                "payment_account_id": "acct_1",
                "amount": 5500.25,
                "currency": "USD",
                "event_time": "2025-01-01T09:00:00",
                "latitude": 38.9,
                "longitude": -77.0,
                "source_channel": "portal",
                "merchant_category": "procurement",
                "full_name": "Jane Doe",
                "email_address": "jane@example.gov",
                "taxpayer_id": "TAX-001",
                "is_fraud": 0,
            }
        ]
    ).write_csv(source)

    summary = run_etl_pipeline(source=source, paths=paths, source_system="treasury", dataset_name="gov_demo")

    silver = pl.read_parquet(summary.silver_path)
    assert summary.rows_extracted == 1
    assert summary.rows_loaded == 1
    assert "tx_id" in silver.columns
    assert "full_name" not in silver.columns
    assert "email_address" not in silver.columns
    assert "full_name_hash" in silver.columns
    assert "email_address_hash" in silver.columns
    assert silver["source_system"][0] == "treasury"
    assert paths.data_path.exists()
    assert paths.feature_store_path.exists()


def test_etl_pipeline_tracks_quality_and_warehouse_status(tmp_path: Path) -> None:
    paths = _etl_paths(tmp_path)
    source = tmp_path / "government_source_quality.csv"
    pl.DataFrame(
        [
            {
                "transaction_id": "dup_tx",
                "beneficiary_id": "case_001",
                "vendor_id": "vendor_001",
                "amount": 100.0,
                "event_time": "2025-01-01T10:00:00",
            },
            {
                "transaction_id": "dup_tx",
                "beneficiary_id": "case_001",
                "vendor_id": "vendor_001",
                "amount": 120.0,
                "event_time": "2025-01-01T11:00:00",
            },
            {
                "transaction_id": "bad_tx",
                "beneficiary_id": "case_002",
                "vendor_id": "vendor_002",
                "amount": -5.0,
                "event_time": "2025-01-01T12:00:00",
            },
        ]
    ).write_csv(source)

    summary = run_etl_pipeline(source=source, paths=paths, source_system="benefits", dataset_name="quality_demo")
    status = list_etl_runs(paths.warehouse_db, limit=5)

    assert summary.rows_extracted == 3
    assert summary.rows_loaded == 1
    assert summary.rows_invalid == 2
    assert summary.duplicates_removed == 1
    assert status[0]["run_id"] == summary.run_id

    conn = duckdb.connect(str(paths.warehouse_db), read_only=True)
    bronze_count = conn.execute("select count(*) from bronze_transactions where etl_run_id = ?", [summary.run_id]).fetchone()[0]
    silver_count = conn.execute("select count(*) from silver_transactions where etl_run_id = ?", [summary.run_id]).fetchone()[0]
    gold_count = conn.execute("select count(*) from gold_features where etl_run_id = ?", [summary.run_id]).fetchone()[0]
    conn.close()

    assert bronze_count == 3
    assert silver_count == 1
    assert gold_count == 1
