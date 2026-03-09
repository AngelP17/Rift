"""Baseline A: Tabular XGBoost classifier for fraud detection."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_PARAMS = {
    "max_depth": 5,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "verbosity": 0,
}


class TabularXGBoost:
    """XGBoost trained on engineered tabular features only."""

    def __init__(self, params: dict | None = None, num_rounds: int = 300):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_rounds = num_rounds
        self.model: xgb.Booster | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:
        pos = y_train.sum()
        neg = len(y_train) - pos
        if pos > 0:
            self.params["scale_pos_weight"] = float(neg / pos)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        log.info("training_xgboost", n_train=len(y_train), rounds=self.num_rounds)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_rounds,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        return {"best_iteration": self.model.best_iteration}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not trained"
        dm = xgb.DMatrix(X)
        return self.model.predict(dm)

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.model_dir / "xgb_tabular.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("model_saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "TabularXGBoost":
        path = path or (cfg.model_dir / "xgb_tabular.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
