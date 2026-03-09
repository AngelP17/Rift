"""DuckDB lakehouse helpers over Parquet artifacts."""

from rift.lakehouse.sql import LakehouseQueryResult, build_default_views, query_lakehouse

__all__ = ["LakehouseQueryResult", "build_default_views", "query_lakehouse"]
