"""Deterministic hashing for replay verification."""

import hashlib
import json
from typing import Any


def canonical_hash(obj: Any) -> str:
    """SHA-256 hash of canonical JSON representation."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]
