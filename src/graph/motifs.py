"""Graph motif and centrality features (optional v1.1 extension)."""

from __future__ import annotations

import numpy as np
import torch

from graph.builder import GraphData
from utils.logging import get_logger

log = get_logger(__name__)


def compute_degree_features(g: GraphData, ntype: str = "transaction") -> np.ndarray:
    """Compute in-degree and out-degree for nodes of the given type."""
    n = g.node_types.get(ntype, 0)
    in_deg = np.zeros(n, dtype=np.float32)
    out_deg = np.zeros(n, dtype=np.float32)

    for etype, ei in g.edge_index.items():
        src_type, _, dst_type = etype
        if src_type == ntype:
            for s in ei[0].tolist():
                if s < n:
                    out_deg[s] += 1
        if dst_type == ntype:
            for d in ei[1].tolist():
                if d < n:
                    in_deg[d] += 1

    return np.stack([in_deg, out_deg], axis=1)


def compute_triangle_count(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Approximate triangle counts per node via neighbor intersection."""
    adj: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
    for s, d in edge_index.t().tolist():
        if s < num_nodes and d < num_nodes:
            adj[s].add(d)
            adj[d].add(s)

    tri = np.zeros(num_nodes, dtype=np.float32)
    for u in range(num_nodes):
        for v in adj[u]:
            common = adj[u].intersection(adj[v])
            tri[u] += len(common)
    tri /= 2.0
    return tri


def compute_motif_features(g: GraphData, ntype: str = "transaction") -> np.ndarray:
    """Combine degree and triangle features into a single motif feature matrix."""
    deg = compute_degree_features(g, ntype)

    from graph.hetero_graph import to_homogeneous_projection
    _, edge_index, _ = to_homogeneous_projection(g, ntype)
    n = g.node_types.get(ntype, 0)
    tri = compute_triangle_count(edge_index, n)

    return np.concatenate([deg, tri.reshape(-1, 1)], axis=1)
