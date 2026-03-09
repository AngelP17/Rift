"""Nearest historical analogs for a given transaction."""

from __future__ import annotations

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


class NearestAnalogFinder:
    """Find the most similar historical transactions by feature distance."""

    def __init__(self, historical_features: np.ndarray, historical_ids: list[str], historical_labels: np.ndarray):
        self.features = historical_features
        self.ids = historical_ids
        self.labels = historical_labels
        self._normalized = None
        self._norms = None

    def _normalize(self):
        if self._normalized is None:
            self._norms = np.linalg.norm(self.features, axis=1, keepdims=True).clip(min=1e-10)
            self._normalized = self.features / self._norms

    def find(self, query: np.ndarray, k: int = 5, fraud_only: bool = False) -> list[dict]:
        """Find k nearest neighbors to the query feature vector."""
        self._normalize()

        if query.ndim == 1:
            query = query.reshape(1, -1)
        q_norm = query / np.linalg.norm(query).clip(min=1e-10)

        sims = (self._normalized @ q_norm.T).squeeze()

        if fraud_only:
            mask = self.labels == 1
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return []
            sub_sims = sims[indices]
            top_k = indices[np.argsort(-sub_sims)[:k]]
        else:
            top_k = np.argsort(-sims)[:k]

        results = []
        for idx in top_k:
            results.append({
                "tx_id": self.ids[idx],
                "similarity": float(sims[idx]),
                "is_fraud": int(self.labels[idx]),
            })
        return results
