from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiftPaths:
    home: Path
    data_path: Path
    feature_store_path: Path
    audit_db: Path
    runs_dir: Path
    datasets_dir: Path
    storage_dir: Path
    lakehouse_dir: Path
    mlflow_dir: Path
    etl_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    lineage_dir: Path
    warehouse_db: Path
    governance_dir: Path
    fairness_dir: Path
    model_cards_dir: Path
    drift_dir: Path
    governance_db: Path
    federated_dir: Path
    query_dir: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_paths() -> RiftPaths:
    home = Path(os.getenv("RIFT_HOME", ".rift"))
    data_path = Path(os.getenv("RIFT_DATA_PATH", home / "data" / "transactions.parquet"))
    feature_store_path = Path(os.getenv("RIFT_FEATURE_STORE_PATH", home / "data" / "features.parquet"))
    audit_db = Path(os.getenv("RIFT_AUDIT_DB", home / "audit" / "rift.duckdb"))
    runs_dir = Path(os.getenv("RIFT_RUNS_DIR", home / "runs"))
    datasets_dir = Path(os.getenv("RIFT_DATASETS_DIR", home / "datasets"))
    storage_dir = Path(os.getenv("RIFT_STORAGE_DIR", home / "storage"))
    lakehouse_dir = Path(os.getenv("RIFT_LAKEHOUSE_DIR", home / "lakehouse"))
    mlflow_dir = Path(os.getenv("RIFT_MLFLOW_DIR", home / "mlruns"))
    etl_dir = Path(os.getenv("RIFT_ETL_DIR", home / "etl"))
    bronze_dir = etl_dir / "bronze"
    silver_dir = etl_dir / "silver"
    gold_dir = etl_dir / "gold"
    lineage_dir = etl_dir / "lineage"
    warehouse_db = Path(os.getenv("RIFT_WAREHOUSE_DB", etl_dir / "warehouse.duckdb"))
    governance_dir = Path(os.getenv("RIFT_GOVERNANCE_DIR", home / "governance"))
    fairness_dir = governance_dir / "fairness"
    model_cards_dir = governance_dir / "model_cards"
    drift_dir = governance_dir / "drift"
    governance_db = Path(os.getenv("RIFT_GOVERNANCE_DB", governance_dir / "governance.duckdb"))
    federated_dir = Path(os.getenv("RIFT_FEDERATED_DIR", home / "federated"))
    query_dir = Path(os.getenv("RIFT_QUERY_DIR", home / "queries"))
    for path in (
        data_path.parent,
        feature_store_path.parent,
        audit_db.parent,
        runs_dir,
        datasets_dir,
        storage_dir,
        lakehouse_dir,
        mlflow_dir,
        bronze_dir,
        silver_dir,
        gold_dir,
        lineage_dir,
        warehouse_db.parent,
        fairness_dir,
        model_cards_dir,
        drift_dir,
        governance_db.parent,
        federated_dir,
        query_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RiftPaths(
        home=home,
        data_path=data_path,
        feature_store_path=feature_store_path,
        audit_db=audit_db,
        runs_dir=runs_dir,
        datasets_dir=datasets_dir,
        storage_dir=storage_dir,
        lakehouse_dir=lakehouse_dir,
        mlflow_dir=mlflow_dir,
        etl_dir=etl_dir,
        bronze_dir=bronze_dir,
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        lineage_dir=lineage_dir,
        warehouse_db=warehouse_db,
        governance_dir=governance_dir,
        fairness_dir=fairness_dir,
        model_cards_dir=model_cards_dir,
        drift_dir=drift_dir,
        governance_db=governance_db,
        federated_dir=federated_dir,
        query_dir=query_dir,
    )
