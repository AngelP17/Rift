"""GraphSAGE encoder for transaction graph embeddings."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import SAGEConv


class FraudGraphSAGE(torch.nn.Module):
    """
    GraphSAGE encoder: 2 layers, hidden 32, output 16.
    Produces transaction node embeddings for hybrid classifier.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
