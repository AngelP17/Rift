"""GAT (Graph Attention Network) encoder for transaction embeddings."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FraudGAT(torch.nn.Module):
    """
    GAT encoder: Multi-head attention for weighted neighbor aggregation.
    Proven lift in fraud graphs (GE-GNN 2025): 2 layers, 8 heads.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 16,
        heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
