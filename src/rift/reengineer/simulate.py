from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from rift.etl.pipeline import run_etl_pipeline
from rift.utils.config import RiftPaths


@dataclass(frozen=True)
class LegacyMigrationSummary:
    output_path: str
    etl_run_id: str
    rows_loaded: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulate_legacy_migration(
    paths: RiftPaths,
    source: Path,
    output_path: Path,
    source_system: str = "legacy_rules_engine",
    sector: str = "fintech",
) -> LegacyMigrationSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    source_str = str(source).replace("'", "''")
    output_str = str(output_path).replace("'", "''")
    conn = duckdb.connect()
    if suffix == ".csv":
        conn.execute(f"copy (select * from read_csv_auto('{source_str}')) to '{output_str}' (format parquet)")
    elif suffix == ".parquet":
        conn.execute(f"copy (select * from read_parquet('{source_str}')) to '{output_str}' (format parquet)")
    else:
        raise ValueError(f"unsupported legacy source format: {source.suffix}")
    conn.close()

    etl_summary = run_etl_pipeline(
        source=output_path,
        paths=paths,
        source_system=source_system,
        dataset_name="legacy_migration",
        sector=sector,
        repo_root=Path(__file__).resolve().parents[3],
    )
    return LegacyMigrationSummary(
        output_path=str(output_path),
        etl_run_id=etl_summary.run_id,
        rows_loaded=etl_summary.rows_loaded,
    )
