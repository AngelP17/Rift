"""Tests for conformal prediction module."""

import numpy as np

from models.conformal import ConformalPredictor, compute_conformal_metrics


class TestConformalPredictor:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(int)

        cp = ConformalPredictor(alpha=0.05)
        cp.fit(scores, labels)

        assert cp.q_hat is not None
        assert cp.q_hat > 0

        preds = cp.predict(scores[:10])
        assert len(preds) == 10
        for p in preds:
            assert "confidence_band" in p
            assert p["confidence_band"] in {"high_confidence_fraud", "review_needed", "high_confidence_legit"}

    def test_predict_bands(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 0, 1, 1])

        cp = ConformalPredictor(alpha=0.1)
        cp.fit(scores, labels)
        bands = cp.predict_bands(scores)
        assert len(bands) == 5

    def test_save_and_load(self, tmp_path):
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])

        cp = ConformalPredictor().fit(scores, labels)
        path = cp.save(tmp_path / "conformal.pkl")

        loaded = ConformalPredictor.load(path)
        assert loaded.q_hat == cp.q_hat


class TestConformalMetrics:
    def test_compute_metrics(self):
        bands = np.array([
            "high_confidence_fraud", "review_needed",
            "high_confidence_legit", "high_confidence_legit",
            "review_needed",
        ])
        labels = np.array([1, 1, 0, 0, 0])

        metrics = compute_conformal_metrics(bands, labels)
        assert "empirical_coverage" in metrics
        assert "average_set_size" in metrics
        assert "review_rate" in metrics
        assert 0 <= metrics["empirical_coverage"] <= 1
        assert metrics["average_set_size"] >= 1
