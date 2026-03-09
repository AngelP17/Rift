"""GAT (Graph Attention Network) encoder for attention-based neighbor aggregation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class GATLayer(nn.Module):
    """Single-head Graph Attention layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.attn = nn.Linear(2 * out_channels, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        Wh = self.W(x)
        src, dst = edge_index[0], edge_index[1]

        edge_h = torch.cat([Wh[src], Wh[dst]], dim=-1)
        e = self.leaky_relu(self.attn(edge_h)).squeeze(-1)

        torch.zeros(x.size(0), dtype=e.dtype, device=e.device)
        e_exp = torch.exp(e - e.max())
        alpha_sum = torch.zeros(x.size(0), device=x.device)
        alpha_sum.index_add_(0, dst, e_exp)
        alpha_norm = e_exp / alpha_sum[dst].clamp(min=1e-10)

        out = torch.zeros_like(Wh)
        out.index_add_(0, dst, alpha_norm.unsqueeze(-1) * Wh[src])
        return out


class FraudGAT(nn.Module):
    """Multi-head GAT encoder for fraud graph embeddings.

    2 layers with multi-head attention, producing fixed-size embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 16,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.heads1 = nn.ModuleList([GATLayer(in_channels, hidden_channels) for _ in range(heads)])
        self.heads2 = nn.ModuleList([GATLayer(hidden_channels * heads, out_channels) for _ in range(1)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_list = [head(x, edge_index) for head in self.heads1]
        h = torch.cat(h_list, dim=-1)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h_list2 = [head(h, edge_index) for head in self.heads2]
        out = h_list2[0] if len(h_list2) == 1 else torch.mean(torch.stack(h_list2), dim=0)
        return out


class GATClassifier(nn.Module):
    """GAT encoder with classification head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        embed_dim: int = 16,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = FraudGAT(in_channels, hidden_channels, embed_dim, heads, dropout)
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
        path = path or (cfg.model_dir / "gat.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: Path, in_channels: int, **kwargs) -> "GATClassifier":
        model = cls(in_channels, **kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
