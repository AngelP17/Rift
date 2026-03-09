from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiftPaths:
    home: Path
    data_path: Path
    audit_db: Path
    runs_dir: Path


def get_paths() -> RiftPaths:
    home = Path(os.getenv("RIFT_HOME", ".rift"))
    data_path = Path(os.getenv("RIFT_DATA_PATH", home / "data" / "transactions.parquet"))
    audit_db = Path(os.getenv("RIFT_AUDIT_DB", home / "audit" / "rift.duckdb"))
    runs_dir = Path(os.getenv("RIFT_RUNS_DIR", home / "runs"))
    for path in (data_path.parent, audit_db.parent, runs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RiftPaths(home=home, data_path=data_path, audit_db=audit_db, runs_dir=runs_dir)
