from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml


DEFAULT_PROFILE = "fintech"


@dataclass(frozen=True)
class SectorProfile:
    name: str
    source_system: str
    field_aliases: dict[str, str]
    privacy_masks: list[str]
    added_columns: dict[str, Any]


def _config_root(repo_root: Path) -> Path:
    return repo_root / "configs" / "sectors"


def available_sector_profiles(repo_root: Path) -> list[str]:
    return sorted(path.stem for path in _config_root(repo_root).glob("*.yaml"))


def load_sector_profile(repo_root: Path, sector: str) -> SectorProfile:
    config_path = _config_root(repo_root) / f"{sector}.yaml"
    if not config_path.exists():
        raise ValueError(f"sector profile not found: {sector}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return SectorProfile(
        name=payload.get("name", sector),
        source_system=payload.get("source_system", sector),
        field_aliases=payload.get("field_aliases", {}),
        privacy_masks=payload.get("privacy_masks", []),
        added_columns=payload.get("added_columns", {}),
    )


def apply_sector_profile(frame: pl.DataFrame, profile: SectorProfile) -> pl.DataFrame:
    renamed = frame
    renames = {
        column: target
        for column, target in profile.field_aliases.items()
        if column in renamed.columns and target not in renamed.columns
    }
    if renames:
        renamed = renamed.rename(renames)

    for column in profile.privacy_masks:
        if column in renamed.columns:
            renamed = renamed.with_columns(pl.lit("[REDACTED]").alias(column))

    for column, value in profile.added_columns.items():
        if column not in renamed.columns:
            renamed = renamed.with_columns(pl.lit(value).alias(column))
    return renamed
