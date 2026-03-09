from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from rift.utils.config import RiftPaths


DEFAULT_VIEW_SQL = {
    "transactions": "create or replace view transactions as select * from read_parquet('{path}')",
    "features": "create or replace view features as select * from read_parquet('{path}')",
    "etl_silver": "create or replace view etl_silver as select * from read_parquet('{path}')",
    "etl_gold": "create or replace view etl_gold as select * from read_parquet('{path}')",
}


@dataclass(frozen=True)
class LakehouseQueryResult:
    sql: str
    rows: int
    columns: list[str]
    preview: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _duckdb_path(paths: RiftPaths) -> Path:
    return paths.lakehouse_dir / "rift_lakehouse.duckdb"


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def build_default_views(paths: RiftPaths) -> Path:
    db_path = _duckdb_path(paths)
    conn = duckdb.connect(str(db_path))
    if paths.data_path.exists():
        conn.execute(DEFAULT_VIEW_SQL["transactions"].format(path=_quote_path(paths.data_path)))
    if paths.feature_store_path.exists():
        conn.execute(DEFAULT_VIEW_SQL["features"].format(path=_quote_path(paths.feature_store_path)))
    silver_files = sorted(paths.silver_dir.glob("*.parquet"))
    gold_files = sorted(paths.gold_dir.glob("*.parquet"))
    if silver_files:
        conn.execute(DEFAULT_VIEW_SQL["etl_silver"].format(path=_quote_path(silver_files[-1])))
    if gold_files:
        conn.execute(DEFAULT_VIEW_SQL["etl_gold"].format(path=_quote_path(gold_files[-1])))
    conn.close()
    return db_path


def query_lakehouse(paths: RiftPaths, sql: str, limit: int = 1000) -> LakehouseQueryResult:
    db_path = build_default_views(paths)
    conn = duckdb.connect(str(db_path), read_only=True)
    query = sql.strip().rstrip(";")
    frame = conn.execute(f"select * from ({query}) limit {limit}").pl()
    conn.close()
    return LakehouseQueryResult(
        sql=sql,
        rows=frame.height,
        columns=frame.columns,
        preview=frame.to_dicts(),
    )
