"""Counterfactual explanation generation."""

from typing import Optional

import numpy as np


def counterfactual_summary(
    features: dict[str, float],
    importance: dict[str, float],
    threshold: float = 0.5,
) -> str:
    """
    Generate counterfactual summary: what would need to change to flip the decision?
    """
    drivers = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    parts = []
    for name, val in drivers:
        if val > 0:  # Pushes toward fraud
            parts.append(f"Lower '{name}' would reduce fraud score")
        else:
            parts.append(f"Higher '{name}' would reduce fraud score")
    return "; ".join(parts) if parts else "No strong drivers identified."


def nearest_analogs(
    query: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float, int]]:
    """Find k nearest historical transactions (index, distance, label)."""
    if len(X_ref) == 0:
        return []
    diffs = np.linalg.norm(X_ref - query, axis=1)
    idx = np.argsort(diffs)[:k]
    return [(int(i), float(diffs[i]), int(y_ref[i])) for i in idx]
