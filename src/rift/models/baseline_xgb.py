from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier


class TabularXGBoostModel:
    def __init__(self) -> None:
        self.model: XGBClassifier | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        positives = max(int(labels.sum()), 1)
        negatives = max(int((1 - labels).sum()), 1)
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=max(negatives / positives, 1.0),
            eval_metric="logloss",
            random_state=7,
        )
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model.predict_proba(features)[:, 1]
