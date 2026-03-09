from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np


def _artifact_bytes(payload: Any) -> int:
    return len(pickle.dumps(payload))


def _downcast_ndarray(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind == "f":
        return array.astype(np.float16)
    return array


def _apply_model_downcast(model: Any) -> None:
    if hasattr(model, "encoder"):
        encoder = model.encoder
        for attr in ("w1", "w2"):
            if hasattr(encoder, attr):
                setattr(encoder, attr, _downcast_ndarray(getattr(encoder, attr)))
    if hasattr(model, "weights"):
        model.weights = _downcast_ndarray(model.weights)
    if hasattr(model, "mean"):
        model.mean = _downcast_ndarray(model.mean)
    if hasattr(model, "std"):
        model.std = _downcast_ndarray(model.std)


def apply_green_optimization(artifact: dict[str, Any], mode: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = (mode or "standard").strip().lower()
    optimized = copy.deepcopy(artifact)
    bytes_before = _artifact_bytes(artifact)
    if normalized == "green":
        _apply_model_downcast(optimized.get("model"))
    bytes_after = _artifact_bytes(optimized)
    metadata = {
        "mode": normalized,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "reduction_ratio": float(1.0 - (bytes_after / bytes_before)) if bytes_before else 0.0,
    }
    optimized["optimization"] = metadata
    return optimized, metadata
