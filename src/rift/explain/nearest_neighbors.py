"""Nearest neighbor utilities for explainability."""

import numpy as np


def k_nearest(
    query: np.ndarray,
    X: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and distances of k nearest neighbors."""
    if metric == "euclidean":
        dist = np.linalg.norm(X - query, axis=1)
    else:
        dist = np.linalg.norm(X - query, axis=1)
    idx = np.argpartition(dist, min(k, len(dist) - 1))[:k]
    idx = idx[np.argsort(dist[idx])]
    return idx, dist[idx]
