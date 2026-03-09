"""Conversion utilities between GraphData and PyG HeteroData."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from graph.builder import GraphData
from utils.logging import get_logger

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

log = get_logger(__name__)


def to_pyg_hetero(g: GraphData) -> "HeteroData":
    """Convert GraphData to torch_geometric HeteroData."""
    from torch_geometric.data import HeteroData

    data = HeteroData()

    for ntype, n in g.node_types.items():
        data[ntype].num_nodes = n
        if ntype in g.node_features:
            data[ntype].x = g.node_features[ntype]

    if g.labels is not None:
        data["transaction"].y = g.labels

    for etype, ei in g.edge_index.items():
        data[etype].edge_index = ei
        if etype in g.edge_attr and g.edge_attr[etype] is not None:
            data[etype].edge_attr = g.edge_attr[etype]

    log.info("converted_to_pyg_hetero")
    return data


def to_homogeneous_projection(g: GraphData, target_ntype: str = "transaction") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create a homogeneous projection on the target node type.

    Produces a simple graph where two target nodes share an edge if they
    share a neighbor in the bipartite graph (e.g., same user or device).

    Returns (x, edge_index, labels).
    """
    n = g.node_types[target_ntype]
    x = g.node_features.get(target_ntype, torch.ones(n, 1))
    labels = g.labels

    neighbor_to_targets: dict[str, dict[int, list[int]]] = {}

    for etype, ei in g.edge_index.items():
        src_type, _, dst_type = etype
        if src_type == target_ntype:
            key = f"{dst_type}"
            if key not in neighbor_to_targets:
                neighbor_to_targets[key] = {}
            for s, d in ei.t().tolist():
                neighbor_to_targets[key].setdefault(d, []).append(s)
        elif dst_type == target_ntype:
            key = f"{src_type}"
            if key not in neighbor_to_targets:
                neighbor_to_targets[key] = {}
            for s, d in ei.t().tolist():
                neighbor_to_targets[key].setdefault(s, []).append(d)

    edges_src, edges_dst = [], []
    for _, groups in neighbor_to_targets.items():
        for _, targets in groups.items():
            if len(targets) > 1:
                for i in range(len(targets)):
                    for j in range(i + 1, min(i + 5, len(targets))):
                        edges_src.append(targets[i])
                        edges_dst.append(targets[j])
                        edges_src.append(targets[j])
                        edges_dst.append(targets[i])

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return x, edge_index, labels
