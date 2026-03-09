"""GraphSAGE/GAT + XGBoost hybrid ensemble."""

from typing import Literal, Optional

import numpy as np
import torch
import xgboost as xgb

from rift.models.baseline_xgb import TabularXGBoost
from rift.models.gat import FraudGAT
from rift.models.graphsage import FraudGraphSAGE


class GraphXGBoostEnsemble:
    """
    Flagship hybrid: GNN embeddings + tabular features -> XGBoost.
    """

    def __init__(
        self,
        encoder_type: Literal["graphsage", "gat"] = "graphsage",
        gnn_hidden: int = 32,
        gnn_out: int = 16,
        xgb_max_depth: int = 5,
        xgb_eta: float = 0.1,
        xgb_subsample: float = 0.8,
        xgb_colsample: float = 0.8,
    ):
        self.encoder_type = encoder_type
        self.gnn_hidden = gnn_hidden
        self.gnn_out = gnn_out
        self.xgb_max_depth = xgb_max_depth
        self.xgb_eta = xgb_eta
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample = xgb_colsample

        self.gnn: Optional[torch.nn.Module] = None
        self.xgb: Optional[xgb.XGBClassifier] = None
        self.feature_cols_: Optional[list[str]] = None
        self.in_channels_: Optional[int] = None

    def _build_encoder(self, in_channels: int) -> torch.nn.Module:
        if self.encoder_type == "graphsage":
            return FraudGraphSAGE(in_channels, self.gnn_hidden, self.gnn_out)
        return FraudGAT(in_channels, self.gnn_hidden, self.gnn_out)

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        if self.gnn is None:
            raise RuntimeError("Ensemble not fitted")
        self.gnn.eval()
        with torch.no_grad():
            emb = self.gnn(x, edge_index)
        return emb.cpu().numpy()

    def fit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "GraphXGBoostEnsemble":
        self.in_channels_ = x.shape[1]
        self.gnn = self._build_encoder(self.in_channels_)
        self.gnn.train()

        opt = torch.optim.Adam(self.gnn.parameters(), lr=0.01)
        for _ in range(50):
            opt.zero_grad()
            out = self.gnn(x, edge_index)
            # Unsupervised or use labels - simple reconstruction / node classification
            # For simplicity: train with BCE if we have labels on nodes
            if y is not None and len(y) == x.shape[0]:
                pred = torch.sigmoid(out.mean(dim=1))
                loss = torch.nn.functional.binary_cross_entropy(
                    pred, torch.tensor(y, dtype=torch.float32)
                )
            else:
                loss = out.pow(2).mean()
            loss.backward()
            opt.step()

        emb = self.get_embeddings(x, edge_index)
        combined = np.hstack([emb, tabular])
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        scale = neg / max(pos, 1)

        self.xgb = xgb.XGBClassifier(
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_eta,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample,
            scale_pos_weight=scale,
            n_estimators=200,
            eval_metric="logloss",
        )
        self.xgb.fit(combined, y)
        self.feature_cols_ = feature_names
        return self

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: np.ndarray,
    ) -> np.ndarray:
        emb = self.get_embeddings(x, edge_index)
        combined = np.hstack([emb, tabular])
        return self.xgb.predict_proba(combined)[:, 1]
