"""I/O utilities for Rift."""

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_json(path: Path | str) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: Path | str) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_parquet(path: Path | str) -> pl.DataFrame:
    """Load Parquet file as Polars DataFrame."""
    return pl.read_parquet(path)


def save_parquet(df: pl.DataFrame, path: Path | str) -> None:
    """Save Polars DataFrame to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def load_csv(path: Path | str) -> pl.DataFrame:
    """Load CSV as Polars DataFrame."""
    return pl.read_csv(path)


def save_csv(df: pl.DataFrame, path: Path | str) -> None:
    """Save Polars DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
