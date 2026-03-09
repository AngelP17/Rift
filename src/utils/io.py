"""I/O utilities for Rift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from utils.config import cfg


def save_parquet(df: pl.DataFrame, name: str, subdir: str = "") -> Path:
    out = cfg.data_dir / subdir / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def load_parquet(name: str, subdir: str = "") -> pl.DataFrame:
    return pl.read_parquet(cfg.data_dir / subdir / f"{name}.parquet")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
