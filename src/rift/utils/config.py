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
    etl_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    lineage_dir: Path
    warehouse_db: Path
    governance_dir: Path
    fairness_dir: Path
    governance_db: Path
    federated_dir: Path


def get_paths() -> RiftPaths:
    home = Path(os.getenv("RIFT_HOME", ".rift"))
    data_path = Path(os.getenv("RIFT_DATA_PATH", home / "data" / "transactions.parquet"))
    feature_store_path = Path(os.getenv("RIFT_FEATURE_STORE_PATH", home / "data" / "features.parquet"))
    audit_db = Path(os.getenv("RIFT_AUDIT_DB", home / "audit" / "rift.duckdb"))
    runs_dir = Path(os.getenv("RIFT_RUNS_DIR", home / "runs"))
    datasets_dir = Path(os.getenv("RIFT_DATASETS_DIR", home / "datasets"))
    etl_dir = Path(os.getenv("RIFT_ETL_DIR", home / "etl"))
    bronze_dir = etl_dir / "bronze"
    silver_dir = etl_dir / "silver"
    gold_dir = etl_dir / "gold"
    lineage_dir = etl_dir / "lineage"
    warehouse_db = Path(os.getenv("RIFT_WAREHOUSE_DB", etl_dir / "warehouse.duckdb"))
    governance_dir = Path(os.getenv("RIFT_GOVERNANCE_DIR", home / "governance"))
    fairness_dir = governance_dir / "fairness"
    governance_db = Path(os.getenv("RIFT_GOVERNANCE_DB", governance_dir / "governance.duckdb"))
    federated_dir = Path(os.getenv("RIFT_FEDERATED_DIR", home / "federated"))
    for path in (
        data_path.parent,
        feature_store_path.parent,
        audit_db.parent,
        runs_dir,
        datasets_dir,
        bronze_dir,
        silver_dir,
        gold_dir,
        lineage_dir,
        warehouse_db.parent,
        fairness_dir,
        governance_db.parent,
        federated_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RiftPaths(
        home=home,
        data_path=data_path,
        feature_store_path=feature_store_path,
        audit_db=audit_db,
        runs_dir=runs_dir,
        datasets_dir=datasets_dir,
        etl_dir=etl_dir,
        bronze_dir=bronze_dir,
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        lineage_dir=lineage_dir,
        warehouse_db=warehouse_db,
        governance_dir=governance_dir,
        fairness_dir=fairness_dir,
        governance_db=governance_db,
        federated_dir=federated_dir,
    )
