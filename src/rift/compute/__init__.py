"""Optional Spark-compatible local compute helpers."""

from rift.compute.spark_compat import spark_available, summarise_parquet_with_spark

__all__ = ["spark_available", "summarise_parquet_with_spark"]
