"""Hybrid ensemble: GNN embeddings + tabular features -> XGBoost/LightGBM."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xgboost as xgb

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class HybridEnsemble:
    """Graph embeddings concatenated with tabular features, fed to a gradient booster."""

    def __init__(
        self,
        gnn_model: torch.nn.Module,
        booster: Literal["xgboost", "lightgbm"] = "xgboost",
        xgb_params: dict | None = None,
        lgb_params: dict | None = None,
        num_rounds: int = 300,
    ):
        self.gnn = gnn_model
        self.booster_type = booster
        self.num_rounds = num_rounds
        self.classifier = None

        self.xgb_params = {
            "max_depth": 5, "eta": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "objective": "binary:logistic",
            "eval_metric": "aucpr", "tree_method": "hist", "verbosity": 0,
            **(xgb_params or {}),
        }
        self.lgb_params = {
            "objective": "binary", "metric": "average_precision",
            "learning_rate": 0.1, "num_leaves": 31, "verbosity": -1,
            **(lgb_params or {}),
        }

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        self.gnn.eval()
        with torch.no_grad():
            if hasattr(self.gnn, "encoder"):
                emb = self.gnn.encoder(x, edge_index)
            else:
                emb = self.gnn(x, edge_index)
        return emb.cpu().numpy()

    def _combine(
        self, x: torch.Tensor, edge_index: torch.Tensor, tabular: np.ndarray | None,
    ) -> np.ndarray:
        emb = self.get_embeddings(x, edge_index)
        if tabular is not None:
            return np.concatenate([emb, tabular], axis=1)
        return emb

    def fit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: np.ndarray,
        tabular: np.ndarray | None = None,
        x_val: torch.Tensor | None = None,
        edge_index_val: torch.Tensor | None = None,
        y_val: np.ndarray | None = None,
        tabular_val: np.ndarray | None = None,
    ) -> dict:
        features = self._combine(x, edge_index, tabular)

        pos = y.sum()
        neg = len(y) - pos
        scale = float(neg / pos) if pos > 0 else 1.0

        log.info("training_hybrid_ensemble", booster=self.booster_type, features=features.shape)

        if self.booster_type == "xgboost":
            self.xgb_params["scale_pos_weight"] = scale
            dtrain = xgb.DMatrix(features, label=y)
            evals = [(dtrain, "train")]
            if x_val is not None and y_val is not None:
                val_feat = self._combine(x_val, edge_index_val, tabular_val)
                dval = xgb.DMatrix(val_feat, label=y_val)
                evals.append((dval, "val"))
            self.classifier = xgb.train(
                self.xgb_params, dtrain,
                num_boost_round=self.num_rounds,
                evals=evals, early_stopping_rounds=30,
                verbose_eval=False,
            )
        else:
            import lightgbm as lgb
            self.lgb_params["scale_pos_weight"] = scale
            dtrain = lgb.Dataset(features, label=y)
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            val_sets = []
            if x_val is not None and y_val is not None:
                val_feat = self._combine(x_val, edge_index_val, tabular_val)
                val_sets = [lgb.Dataset(val_feat, label=y_val, reference=dtrain)]
            self.classifier = lgb.train(
                self.lgb_params, dtrain,
                num_boost_round=self.num_rounds,
                valid_sets=val_sets,
                callbacks=callbacks,
            )

        return {"booster": self.booster_type, "features_dim": features.shape[1]}

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor, tabular: np.ndarray | None = None,
    ) -> np.ndarray:
        features = self._combine(x, edge_index, tabular)
        if self.booster_type == "xgboost":
            dm = xgb.DMatrix(features)
            return self.classifier.predict(dm)
        else:
            return self.classifier.predict(features)

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.model_dir / f"hybrid_{self.booster_type}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("ensemble_saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Path) -> "HybridEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)
