"""Centralized configuration for Rift."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RiftConfig:
    seed: int = int(os.getenv("RIFT_SEED", "42"))
    log_level: str = os.getenv("RIFT_LOG_LEVEL", "INFO")
    audit_db: Path = field(default_factory=lambda: _ROOT / os.getenv("RIFT_AUDIT_DB", "data/audit.duckdb"))
    model_dir: Path = field(default_factory=lambda: _ROOT / os.getenv("RIFT_MODEL_DIR", "artifacts/models"))
    data_dir: Path = field(default_factory=lambda: _ROOT / os.getenv("RIFT_DATA_DIR", "data"))
    device: str = os.getenv("RIFT_DEVICE", "cpu")

    def ensure_dirs(self) -> None:
        for d in (self.audit_db.parent, self.model_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)


cfg = RiftConfig()
