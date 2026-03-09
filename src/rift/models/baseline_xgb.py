"""Tabular XGBoost baseline for fraud detection."""

from typing import Optional

import numpy as np
import xgboost as xgb

from rift.data.schemas import FEATURE_COLUMNS


class TabularXGBoost:
    """XGBoost classifier on tabular features only. Sanity baseline."""

    def __init__(
        self,
        max_depth: int = 5,
        eta: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.eta = eta
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols_: Optional[list[str]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "TabularXGBoost":
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        scale = neg / max(pos, 1) if self.scale_pos_weight is None else self.scale_pos_weight

        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.eta,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.model.fit(X, y)
        self.feature_cols_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)[:, 1]  # Fraud probability

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)
