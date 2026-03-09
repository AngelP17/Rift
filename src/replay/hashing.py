"""Deterministic hashing for decision integrity verification."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: dict[str, Any]) -> str:
    """Produce canonical JSON (sorted keys, no extra whitespace)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def decision_hash(payload: dict[str, Any]) -> str:
    """SHA-256 hash of the canonical JSON representation of a decision payload."""
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_hash(payload: dict[str, Any], expected_hash: str) -> bool:
    """Verify that a payload matches its expected hash."""
    return decision_hash(payload) == expected_hash
