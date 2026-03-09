from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from rift.graph.builder import TransactionGraph


def _aggregate(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros_like(features)
    src, dst = edge_index
    out = np.zeros_like(features, dtype=float)
    counts = np.zeros((features.shape[0], 1), dtype=float)
    np.add.at(out, dst, features[src])
    np.add.at(counts, dst, 1.0)
    counts[counts == 0] = 1.0
    return out / counts


@dataclass
class GraphSAGEEncoder:
    in_dim: int
    hidden_dim: int = 32
    out_dim: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 0.15, size=(self.in_dim * 2, self.hidden_dim))
        self.w2 = rng.normal(0.0, 0.15, size=(self.hidden_dim * 2, self.out_dim))

    def encode(self, features: np.ndarray, graph: TransactionGraph) -> np.ndarray:
        h0 = np.asarray(features, dtype=float)
        n1 = _aggregate(h0, graph.edge_index)
        h1 = np.tanh(np.concatenate([h0, n1], axis=1) @ self.w1)
        n2 = _aggregate(h1, graph.edge_index)
        h2 = np.tanh(np.concatenate([h1, n2], axis=1) @ self.w2)
        return h2


class GraphSAGEOnlyModel:
    def __init__(self, encoder: GraphSAGEEncoder) -> None:
        self.encoder = encoder
        self.classifier = LogisticRegression(max_iter=1_000, class_weight="balanced")

    def fit(self, features: np.ndarray, graph: TransactionGraph, labels: np.ndarray) -> None:
        embeddings = self.encoder.encode(features, graph)
        self.classifier.fit(embeddings, labels)

    def predict_proba(self, features: np.ndarray, graph: TransactionGraph) -> np.ndarray:
        embeddings = self.encoder.encode(features, graph)
        return self.classifier.predict_proba(embeddings)[:, 1]

    def explain_weights(self) -> np.ndarray:
        return self.classifier.coef_[0]
