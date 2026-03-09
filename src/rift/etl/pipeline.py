from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from rift.adapters.sectors import DEFAULT_PROFILE, apply_sector_profile, load_sector_profile
from rift.features.engine import build_features
from rift.storage.backends import get_storage_backend
from rift.utils.config import RiftPaths
from rift.utils.io import write_json


CANONICAL_COLUMNS = [
    "tx_id",
    "user_id",
    "merchant_id",
    "device_id",
    "account_id",
    "amount",
    "currency",
    "timestamp",
    "lat",
    "lon",
    "channel",
    "mcc",
    "is_fraud",
]

COLUMN_ALIASES = {
    "transaction_id": "tx_id",
    "txn_id": "tx_id",
    "payment_id": "tx_id",
    "record_id": "tx_id",
    "beneficiary_id": "user_id",
    "customer_id": "user_id",
    "citizen_id": "user_id",
    "vendor_id": "merchant_id",
    "supplier_id": "merchant_id",
    "payee_id": "merchant_id",
    "payment_account_id": "account_id",
    "program_account_id": "account_id",
    "endpoint_id": "device_id",
    "instrument_id": "device_id",
    "event_time": "timestamp",
    "payment_time": "timestamp",
    "transaction_time": "timestamp",
    "latitude": "lat",
    "longitude": "lon",
    "source_channel": "channel",
    "merchant_category": "mcc",
    "category": "mcc",
    "fraud_label": "is_fraud",
    "label": "is_fraud",
}

DEFAULTS: dict[str, Any] = {
    "user_id": "unknown_user",
    "merchant_id": "unknown_merchant",
    "device_id": "unknown_device",
    "account_id": "unknown_account",
    "currency": "USD",
    "lat": 0.0,
    "lon": 0.0,
    "channel": "government_batch",
    "mcc": "public_services",
    "is_fraud": 0,
}

SENSITIVE_COLUMNS = {
    "full_name",
    "name",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "ssn",
    "national_id",
    "taxpayer_id",
    "tax_id",
    "address",
}


@dataclass(frozen=True)
class ETLRunSummary:
    run_id: str
    dataset_name: str
    source_system: str
    source_path: str
    rows_extracted: int
    rows_valid: int
    rows_invalid: int
    rows_loaded: int
    duplicates_removed: int
    bronze_path: str
    silver_path: str
    gold_path: str
    manifest_path: str
    warehouse_db: str
    bronze_object: str | None = None
    silver_object: str | None = None
    gold_object: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonicalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _read_source(source: Path) -> pl.DataFrame:
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(source, try_parse_dates=True)
    if suffix == ".json":
        return pl.read_json(source)
    if suffix == ".parquet":
        return pl.read_parquet(source)
    raise ValueError(f"unsupported source format: {source.suffix}")


def _hash_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_hash(row: dict[str, Any]) -> str:
    payload = "|".join(str(row.get(column, "")) for column in CANONICAL_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _normalize_columns(frame: pl.DataFrame) -> pl.DataFrame:
    renamed = frame.rename({column: _canonicalize_name(column) for column in frame.columns})
    alias_renames = {
        column: COLUMN_ALIASES[column]
        for column in renamed.columns
        if column in COLUMN_ALIASES and COLUMN_ALIASES[column] not in renamed.columns
    }
    if alias_renames:
        renamed = renamed.rename(alias_renames)
    return renamed


def _pseudonymous_user_id(frame: pl.DataFrame) -> pl.Expr:
    sources = [column for column in ("taxpayer_id", "tax_id", "national_id") if column in frame.columns]
    if not sources:
        return pl.lit(None, dtype=pl.Utf8)
    first_available = pl.coalesce([pl.col(column).cast(pl.Utf8) for column in sources])
    return first_available.map_elements(
        lambda value: f"user_{_hash_text(value)[:12]}" if _hash_text(value) else None,
        return_dtype=pl.Utf8,
    )


def _prepare_silver_frame(raw: pl.DataFrame, run_id: str, source_system: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    normalized = _normalize_columns(raw).with_row_index("source_row_number")
    source_columns = list(normalized.columns)

    if "user_id" not in normalized.columns:
        normalized = normalized.with_columns(_pseudonymous_user_id(normalized).alias("user_id"))

    for column, default in DEFAULTS.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(default).alias(column))

    if "tx_id" not in normalized.columns:
        normalized = normalized.with_columns(
            (pl.lit(f"{source_system}_") + pl.col("source_row_number").cast(pl.Utf8)).alias("tx_id")
        )

    for column in SENSITIVE_COLUMNS.intersection(set(normalized.columns)):
        normalized = normalized.with_columns(
            pl.col(column).cast(pl.Utf8).map_elements(_hash_text, return_dtype=pl.Utf8).alias(f"{column}_hash")
        )

    casted = normalized.with_columns(
        pl.col("tx_id").cast(pl.Utf8),
        pl.col("user_id").cast(pl.Utf8).fill_null(DEFAULTS["user_id"]),
        pl.col("merchant_id").cast(pl.Utf8).fill_null(DEFAULTS["merchant_id"]),
        pl.col("device_id").cast(pl.Utf8).fill_null(DEFAULTS["device_id"]),
        pl.col("account_id").cast(pl.Utf8).fill_null(DEFAULTS["account_id"]),
        pl.col("amount").cast(pl.Float64, strict=False),
        pl.col("currency").cast(pl.Utf8).fill_null(DEFAULTS["currency"]),
        pl.col("timestamp").map_elements(_coerce_timestamp, return_dtype=pl.Datetime),
        pl.col("lat").cast(pl.Float64, strict=False).fill_null(float(DEFAULTS["lat"])),
        pl.col("lon").cast(pl.Float64, strict=False).fill_null(float(DEFAULTS["lon"])),
        pl.col("channel").cast(pl.Utf8).fill_null(DEFAULTS["channel"]),
        pl.col("mcc").cast(pl.Utf8).fill_null(DEFAULTS["mcc"]),
        pl.col("is_fraud").cast(pl.Int64, strict=False).fill_null(int(DEFAULTS["is_fraud"])),
    )

    valid = casted.filter(
        pl.col("tx_id").is_not_null()
        & pl.col("amount").is_not_null()
        & (pl.col("amount") >= 0.0)
        & pl.col("timestamp").is_not_null()
    )
    before_dedup = valid.height
    valid = valid.sort("timestamp").unique(subset=["tx_id"], keep="last", maintain_order=True)
    duplicates_removed = before_dedup - valid.height

    sensitive_to_drop = [column for column in valid.columns if column in SENSITIVE_COLUMNS]
    if sensitive_to_drop:
        valid = valid.drop(sensitive_to_drop)

    ingested_at = datetime.now(timezone.utc).isoformat()
    valid = valid.with_columns(
        pl.lit(run_id).alias("etl_run_id"),
        pl.lit(source_system).alias("source_system"),
        pl.lit(ingested_at).alias("ingested_at"),
    )

    valid = valid.with_columns(
        pl.struct(CANONICAL_COLUMNS).map_elements(_record_hash, return_dtype=pl.Utf8).alias("record_hash")
    )

    ordered_columns = CANONICAL_COLUMNS + [
        column
        for column in valid.columns
        if column not in CANONICAL_COLUMNS and column not in {"source_row_number"}
    ]
    silver = valid.select(ordered_columns)

    quality_report = {
        "source_columns": source_columns,
        "silver_columns": silver.columns,
        "rows_extracted": raw.height,
        "rows_valid": silver.height,
        "rows_invalid": raw.height - silver.height,
        "duplicates_removed": duplicates_removed,
        "sensitive_columns_redacted": sorted(SENSITIVE_COLUMNS.intersection(set(raw.columns))),
        "null_counts": {column: int(silver[column].null_count()) for column in CANONICAL_COLUMNS},
    }
    return silver, quality_report


def _write_current_snapshots(paths: RiftPaths, silver: pl.DataFrame, gold: pl.DataFrame) -> None:
    silver.write_parquet(paths.data_path)
    gold.write_parquet(paths.feature_store_path)


def _ensure_table(conn: duckdb.DuckDBPyConnection, table_name: str, parquet_path: Path) -> None:
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = ?",
        [table_name],
    ).fetchone()[0]
    parquet_columns = pl.read_parquet(parquet_path).columns
    if exists:
        current_columns = [row[1] for row in conn.execute(f"pragma table_info('{table_name}')").fetchall()]
        if current_columns == parquet_columns:
            return
        conn.execute(f"drop table {table_name}")
    conn.execute(
        f"create table {table_name} as select * from read_parquet(?) limit 0",
        [str(parquet_path)],
    )


def _load_parquet_table(conn: duckdb.DuckDBPyConnection, table_name: str, parquet_path: Path, run_id: str) -> None:
    _ensure_table(conn, table_name, parquet_path)
    conn.execute(f"delete from {table_name} where etl_run_id = ?", [run_id])
    conn.execute(f"insert into {table_name} by name select * from read_parquet(?)", [str(parquet_path)])


def _load_bronze_rows(conn: duckdb.DuckDBPyConnection, bronze_rows: pl.DataFrame, run_id: str) -> None:
    existing = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'bronze_transactions'"
    ).fetchone()[0]
    expected_columns = ["etl_run_id", "source_system", "source_path", "source_row_number", "raw_record_json"]
    if existing:
        current_columns = [
            row[1]
            for row in conn.execute("pragma table_info('bronze_transactions')").fetchall()
        ]
        if current_columns != expected_columns:
            conn.execute("drop table bronze_transactions")
    conn.execute(
        """
        create table if not exists bronze_transactions (
            etl_run_id varchar,
            source_system varchar,
            source_path varchar,
            source_row_number bigint,
            raw_record_json varchar
        )
        """
    )
    conn.execute("delete from bronze_transactions where etl_run_id = ?", [run_id])
    conn.register("bronze_rows_view", bronze_rows.to_arrow())
    conn.execute("insert into bronze_transactions by name select * from bronze_rows_view")
    conn.unregister("bronze_rows_view")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def run_etl_pipeline(
    source: Path,
    paths: RiftPaths,
    source_system: str = "government_finance",
    dataset_name: str = "transactions",
    sector: str = DEFAULT_PROFILE,
    repo_root: Path | None = None,
) -> ETLRunSummary:
    raw = _read_source(source)
    resolved_root = repo_root or Path(__file__).resolve().parents[3]
    raw = apply_sector_profile(raw, load_sector_profile(resolved_root, sector))
    storage = get_storage_backend(paths)
    run_id = f"etl_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    bronze_path = paths.bronze_dir / f"{run_id}_{dataset_name}.parquet"
    silver_path = paths.silver_dir / f"{run_id}_{dataset_name}.parquet"
    gold_path = paths.gold_dir / f"{run_id}_{dataset_name}_features.parquet"
    manifest_path = paths.lineage_dir / f"{run_id}_{dataset_name}.json"

    bronze = raw.with_columns(
        pl.lit(run_id).alias("etl_run_id"),
        pl.lit(source_system).alias("source_system"),
        pl.lit(str(source)).alias("source_path"),
    )
    bronze.write_parquet(bronze_path)
    bronze_rows = raw.with_row_index("source_row_number").with_columns(
        pl.lit(run_id).alias("etl_run_id"),
        pl.lit(source_system).alias("source_system"),
        pl.lit(str(source)).alias("source_path"),
    )
    bronze_rows = bronze_rows.with_columns(
        pl.struct(pl.exclude(["source_row_number", "etl_run_id", "source_system", "source_path"]))
        .map_elements(lambda row: json.dumps(row, sort_keys=True, default=str), return_dtype=pl.Utf8)
        .alias("raw_record_json")
    ).select("etl_run_id", "source_system", "source_path", "source_row_number", "raw_record_json")

    silver, quality = _prepare_silver_frame(raw, run_id=run_id, source_system=source_system)
    gold = build_features(silver)
    gold.write_parquet(gold_path)
    silver.write_parquet(silver_path)
    _write_current_snapshots(paths, silver, gold)
    bronze_object = storage.save_parquet(bronze, f"etl/bronze/{bronze_path.name}")
    silver_object = storage.save_parquet(silver, f"etl/silver/{silver_path.name}")
    gold_object = storage.save_parquet(gold, f"etl/gold/{gold_path.name}")

    conn = duckdb.connect(str(paths.warehouse_db))
    conn.execute(
        """
        create table if not exists etl_runs (
            run_id varchar primary key,
            dataset_name varchar,
            source_system varchar,
            source_path varchar,
            rows_extracted bigint,
            rows_valid bigint,
            rows_invalid bigint,
            rows_loaded bigint,
            duplicates_removed bigint,
            bronze_path varchar,
            silver_path varchar,
            gold_path varchar,
            manifest_path varchar,
            created_at timestamp
        )
        """
    )
    _load_bronze_rows(conn, bronze_rows, run_id)
    _load_parquet_table(conn, "silver_transactions", silver_path, run_id)
    _load_parquet_table(conn, "gold_features", gold_path, run_id)

    rows_loaded = silver.height
    summary = ETLRunSummary(
        run_id=run_id,
        dataset_name=dataset_name,
        source_system=source_system,
        source_path=str(source),
        rows_extracted=raw.height,
        rows_valid=quality["rows_valid"],
        rows_invalid=quality["rows_invalid"],
        rows_loaded=rows_loaded,
        duplicates_removed=quality["duplicates_removed"],
        bronze_path=str(bronze_path),
        silver_path=str(silver_path),
        gold_path=str(gold_path),
        manifest_path=str(manifest_path),
        warehouse_db=str(paths.warehouse_db),
        bronze_object=bronze_object,
        silver_object=silver_object,
        gold_object=gold_object,
    )

    manifest = {
        "summary": summary.to_dict(),
        "quality": quality,
        "artifacts": {
            "bronze": str(bronze_path),
            "silver": str(silver_path),
            "gold": str(gold_path),
            "current_data_snapshot": str(paths.data_path),
            "current_feature_snapshot": str(paths.feature_store_path),
        },
    }
    _write_manifest(manifest_path, manifest)

    conn.execute(
        """
        insert or replace into etl_runs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            summary.run_id,
            summary.dataset_name,
            summary.source_system,
            summary.source_path,
            summary.rows_extracted,
            summary.rows_valid,
            summary.rows_invalid,
            summary.rows_loaded,
            summary.duplicates_removed,
            summary.bronze_path,
            summary.silver_path,
            summary.gold_path,
            summary.manifest_path,
            datetime.now(timezone.utc).replace(tzinfo=None),
        ],
    )
    conn.close()
    return summary


def list_etl_runs(warehouse_db: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not warehouse_db.exists():
        return []
    conn = duckdb.connect(str(warehouse_db), read_only=True)
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'etl_runs'"
    ).fetchone()[0]
    if not exists:
        conn.close()
        return []
    rows = conn.execute(
        """
        select run_id, dataset_name, source_system, rows_extracted, rows_valid, rows_invalid,
               rows_loaded, duplicates_removed, source_path, bronze_path, silver_path, gold_path,
               manifest_path, created_at
        from etl_runs
        order by created_at desc
        limit ?
        """,
        [limit],
    ).fetchall()
    conn.close()
    keys = [
        "run_id",
        "dataset_name",
        "source_system",
        "rows_extracted",
        "rows_valid",
        "rows_invalid",
        "rows_loaded",
        "duplicates_removed",
        "source_path",
        "bronze_path",
        "silver_path",
        "gold_path",
        "manifest_path",
        "created_at",
    ]
    return [dict(zip(keys, row, strict=False)) for row in rows]
