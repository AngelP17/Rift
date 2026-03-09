from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from rift.graph.builder import TransactionGraph
from rift.models.graphsage import GraphSAGEEncoder


class GraphHybridModel:
    def __init__(self, encoder: GraphSAGEEncoder) -> None:
        self.encoder = encoder
        self.model: XGBClassifier | None = None

    def fit(self, features: np.ndarray, graph: TransactionGraph, labels: np.ndarray) -> None:
        embeddings = self.encoder.encode(features, graph)
        fused = np.concatenate([features, embeddings], axis=1)
        positives = max(int(labels.sum()), 1)
        negatives = max(int((1 - labels).sum()), 1)
        self.model = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=max(negatives / positives, 1.0),
            eval_metric="logloss",
            random_state=7,
        )
        self.model.fit(fused, labels)

    def predict_proba(self, features: np.ndarray, graph: TransactionGraph) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        embeddings = self.encoder.encode(features, graph)
        fused = np.concatenate([features, embeddings], axis=1)
        return self.model.predict_proba(fused)[:, 1]

    def explain_contributions(self, features: np.ndarray, graph: TransactionGraph) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        embeddings = self.encoder.encode(features, graph)
        fused = np.concatenate([features, embeddings], axis=1)
        weights = np.asarray(self.model.feature_importances_, dtype=float)
        return fused * weights
