"""Baseline B & Encoder: GraphSAGE for transaction-level embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE encoder producing node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.neigh_lin1 = nn.Linear(in_channels, hidden_channels)
        self.neigh_lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def _sage_conv(
        self, x: torch.Tensor, edge_index: torch.Tensor, lin_self: nn.Linear, lin_neigh: nn.Linear,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        neigh_sum = torch.zeros(num_nodes, x.size(1), device=x.device)
        neigh_count = torch.zeros(num_nodes, 1, device=x.device)
        neigh_sum.index_add_(0, dst, x[src])
        ones = torch.ones(src.size(0), 1, device=x.device)
        neigh_count.index_add_(0, dst, ones)
        neigh_mean = neigh_sum / neigh_count.clamp(min=1)

        out = lin_self(x) + lin_neigh(neigh_mean)
        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self._sage_conv(x, edge_index, self.lin1, self.neigh_lin1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self._sage_conv(h, edge_index, self.lin2, self.neigh_lin2)
        return h


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE encoder with MLP head for direct classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        embed_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, embed_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x, edge_index)
        return self.head(emb).squeeze(-1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x, edge_index)

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.model_dir / "graphsage.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        log.info("model_saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Path, in_channels: int, **kwargs) -> "GraphSAGEClassifier":
        model = cls(in_channels, **kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
