"""Tests for graph construction."""


from graph.builder import GraphData, build_graph


class TestGraphBuilder:
    def test_builds_graph(self, small_dataset):
        g = build_graph(small_dataset)
        assert isinstance(g, GraphData)
        assert g.node_types["transaction"] == len(small_dataset)

    def test_has_all_node_types(self, small_dataset):
        g = build_graph(small_dataset)
        for ntype in ["user", "merchant", "device", "account", "transaction"]:
            assert ntype in g.node_types
            assert g.node_types[ntype] > 0

    def test_has_edges(self, small_dataset):
        g = build_graph(small_dataset)
        assert len(g.edge_index) > 0
        for etype, ei in g.edge_index.items():
            assert ei.shape[0] == 2
            assert ei.shape[1] > 0

    def test_has_labels(self, small_dataset):
        g = build_graph(small_dataset)
        assert g.labels is not None
        assert len(g.labels) == len(small_dataset)

    def test_with_feature_cols(self, featured_dataset):
        from features.engine import FEATURE_COLUMNS
        g = build_graph(featured_dataset, feature_cols=FEATURE_COLUMNS)
        tx_feats = g.node_features["transaction"]
        assert tx_feats.shape[0] == len(featured_dataset)
        assert tx_feats.shape[1] > 1

    def test_homogeneous_projection(self, small_dataset):
        from graph.hetero_graph import to_homogeneous_projection
        g = build_graph(small_dataset)
        x, edge_index, labels = to_homogeneous_projection(g)
        assert x.shape[0] == g.node_types["transaction"]
        assert edge_index.shape[0] == 2
