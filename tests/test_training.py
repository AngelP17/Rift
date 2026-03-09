"""Tests for the training pipeline."""

import numpy as np

from models.baseline_xgb import TabularXGBoost
from models.gat import FraudGAT, GATClassifier
from models.graphsage import GraphSAGEClassifier, GraphSAGEEncoder


class TestTabularXGBoost:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 10)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        model = TabularXGBoost(num_rounds=10)
        model.fit(X[:160], y[:160], X[160:], y[160:])
        preds = model.predict_proba(X[160:])

        assert len(preds) == 40
        assert all(0 <= p <= 1 for p in preds)

    def test_save_and_load(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        model = TabularXGBoost(num_rounds=5)
        model.fit(X, y)
        path = model.save(tmp_path / "test_xgb.pkl")

        loaded = TabularXGBoost.load(path)
        p1 = model.predict_proba(X[:10])
        p2 = loaded.predict_proba(X[:10])
        np.testing.assert_array_almost_equal(p1, p2)


class TestGraphSAGE:
    def test_encoder_forward(self):
        import torch
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        encoder = GraphSAGEEncoder(in_channels=10)
        out = encoder(x, edge_index)
        assert out.shape == (50, 16)

    def test_classifier_forward(self):
        import torch
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        model = GraphSAGEClassifier(in_channels=10)
        out = model(x, edge_index)
        assert out.shape == (50,)

    def test_get_embeddings(self):
        import torch
        x = torch.randn(30, 8)
        edge_index = torch.randint(0, 30, (2, 60))
        model = GraphSAGEClassifier(in_channels=8, embed_dim=16)
        emb = model.get_embeddings(x, edge_index)
        assert emb.shape == (30, 16)


class TestGAT:
    def test_encoder_forward(self):
        import torch
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        encoder = FraudGAT(in_channels=10)
        out = encoder(x, edge_index)
        assert out.shape == (50, 16)

    def test_classifier_forward(self):
        import torch
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        model = GATClassifier(in_channels=10)
        out = model(x, edge_index)
        assert out.shape == (50,)
