"""Configuration management for Rift."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = Path(os.getenv("RIFT_ARTIFACTS_DIR", str(WORKSPACE_ROOT / "artifacts")))
MODEL_DIR = Path(os.getenv("RIFT_MODEL_DIR", str(ARTIFACTS_DIR / "models")))
AUDIT_DB_PATH = Path(os.getenv("RIFT_AUDIT_DB", str(ARTIFACTS_DIR / "audit.duckdb")))
MLFLOW_URI = os.getenv("RIFT_MLFLOW_URI", str(ARTIFACTS_DIR / "mlruns"))

# Data
SEED = int(os.getenv("RIFT_SEED", "42"))
DEFAULT_FRAUD_RATE = float(os.getenv("RIFT_FRAUD_RATE", "0.02"))

# API
API_HOST = os.getenv("RIFT_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RIFT_API_PORT", "8000"))

# Logging
LOG_LEVEL = os.getenv("RIFT_LOG_LEVEL", "INFO")


def ensure_dirs() -> None:
    """Ensure artifact directories exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
