from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from rift.etl.pipeline import run_etl_pipeline
from rift.utils.config import RiftPaths
from rift.utils.io import write_json


SUPPORTED_ADAPTERS = ("ieee_cis", "credit_card_fraud")


@dataclass(frozen=True)
class DatasetPrepareSummary:
    dataset_id: str
    adapter: str
    source_path: str
    canonical_path: str
    manifest_path: str
    rows_prepared: int
    auto_etl_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_source(source: Path) -> pl.DataFrame:
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(source, try_parse_dates=True)
    if suffix == ".parquet":
        return pl.read_parquet(source)
    if suffix == ".json":
        return pl.read_json(source)
    raise ValueError(f"unsupported dataset source format: {source.suffix}")


def _ieee_cis_adapter(frame: pl.DataFrame) -> pl.DataFrame:
    base_time = datetime(2017, 12, 1, tzinfo=timezone.utc)
    working = frame.with_row_index("row_idx")
    tx_time = (
        pl.lit(base_time)
        + pl.duration(seconds=pl.col("TransactionDT").cast(pl.Int64, strict=False).fill_null(0))
    ).alias("timestamp")
    return working.select(
        pl.col("TransactionID").cast(pl.Utf8).alias("tx_id"),
        pl.coalesce([pl.col("card1").cast(pl.Utf8), pl.col("row_idx").cast(pl.Utf8)]).alias("user_id"),
        pl.coalesce([pl.col("ProductCD").cast(pl.Utf8), pl.lit("ieee_product")]).alias("merchant_id"),
        pl.coalesce([pl.col("DeviceInfo").cast(pl.Utf8), pl.col("DeviceType").cast(pl.Utf8), pl.lit("unknown_device")]).alias("device_id"),
        (
            pl.coalesce([pl.col("card1").cast(pl.Utf8), pl.lit("card")])
            + "_"
            + pl.coalesce([pl.col("card4").cast(pl.Utf8), pl.lit("na")])
        ).alias("account_id"),
        pl.col("TransactionAmt").cast(pl.Float64, strict=False).fill_null(0.0).alias("amount"),
        pl.lit("USD").alias("currency"),
        tx_time,
        pl.col("addr1").cast(pl.Float64, strict=False).fill_null(0.0).alias("lat"),
        pl.col("addr2").cast(pl.Float64, strict=False).fill_null(0.0).alias("lon"),
        pl.coalesce([pl.col("DeviceType").cast(pl.Utf8), pl.lit("online")]).alias("channel"),
        pl.coalesce([pl.col("ProductCD").cast(pl.Utf8), pl.lit("retail")]).alias("mcc"),
        pl.col("isFraud").cast(pl.Int64, strict=False).fill_null(0).alias("is_fraud"),
    )


def _credit_card_adapter(frame: pl.DataFrame) -> pl.DataFrame:
    base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    working = frame.with_row_index("row_idx")
    user_bucket = (
        pl.concat_str(
            [
                pl.col("V1").cast(pl.Float64, strict=False).round(2).cast(pl.Utf8).fill_null("0"),
                pl.col("V2").cast(pl.Float64, strict=False).round(2).cast(pl.Utf8).fill_null("0"),
            ],
            separator="_",
        )
        .alias("user_bucket")
    )
    with_bucket = working.with_columns(user_bucket)
    return with_bucket.select(
        ("ccf_" + pl.col("row_idx").cast(pl.Utf8)).alias("tx_id"),
        ("user_" + pl.col("user_bucket")).alias("user_id"),
        pl.lit("credit_card_network").alias("merchant_id"),
        pl.lit("card_terminal").alias("device_id"),
        ("acct_" + pl.col("user_bucket")).alias("account_id"),
        pl.col("Amount").cast(pl.Float64, strict=False).fill_null(0.0).alias("amount"),
        pl.lit("EUR").alias("currency"),
        (pl.lit(base_time) + pl.duration(seconds=pl.col("Time").cast(pl.Int64, strict=False).fill_null(0))).alias("timestamp"),
        pl.lit(0.0).alias("lat"),
        pl.lit(0.0).alias("lon"),
        pl.lit("card_present").alias("channel"),
        pl.lit("consumer_credit").alias("mcc"),
        pl.col("Class").cast(pl.Int64, strict=False).fill_null(0).alias("is_fraud"),
    )


ADAPTERS = {
    "ieee_cis": _ieee_cis_adapter,
    "credit_card_fraud": _credit_card_adapter,
}


def prepare_public_dataset(
    source: Path,
    adapter: str,
    paths: RiftPaths,
    auto_etl: bool = True,
) -> DatasetPrepareSummary:
    if adapter not in ADAPTERS:
        raise ValueError(f"unsupported adapter '{adapter}'. Supported adapters: {', '.join(SUPPORTED_ADAPTERS)}")

    frame = _read_source(source)
    canonical = ADAPTERS[adapter](frame).sort("timestamp")
    dataset_id = f"{adapter}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    canonical_path = paths.datasets_dir / f"{dataset_id}.parquet"
    manifest_path = paths.datasets_dir / f"{dataset_id}.json"
    canonical.write_parquet(canonical_path)

    etl_run_id: str | None = None
    if auto_etl:
        etl_summary = run_etl_pipeline(
            source=canonical_path,
            paths=paths,
            source_system=f"public_dataset_{adapter}",
            dataset_name=adapter,
        )
        etl_run_id = etl_summary.run_id

    summary = DatasetPrepareSummary(
        dataset_id=dataset_id,
        adapter=adapter,
        source_path=str(source),
        canonical_path=str(canonical_path),
        manifest_path=str(manifest_path),
        rows_prepared=canonical.height,
        auto_etl_run_id=etl_run_id,
    )
    write_json(
        manifest_path,
        {
            "summary": summary.to_dict(),
            "supported_adapters": list(SUPPORTED_ADAPTERS),
            "canonical_columns": canonical.columns,
        },
    )
    return summary


def list_prepared_datasets(paths: RiftPaths, limit: int = 10) -> list[dict[str, Any]]:
    manifests = sorted(paths.datasets_dir.glob("*.json"), reverse=True)[:limit]
    output: list[dict[str, Any]] = []
    for manifest in manifests:
        output.append(json_load(manifest))
    return output


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
