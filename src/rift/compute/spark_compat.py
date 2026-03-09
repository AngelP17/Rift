from __future__ import annotations

from pathlib import Path


def spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except Exception:
        return False


def summarise_parquet_with_spark(path: Path) -> dict[str, object]:
    if not spark_available():
        raise RuntimeError("pyspark is not installed")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("RiftLocalSpark")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    df = spark.read.parquet(str(path))
    summary = {
        "rows": df.count(),
        "columns": df.columns,
        "schema": df.schema.simpleString(),
    }
    spark.stop()
    return summary
