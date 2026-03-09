"""Conformal prediction layer for uncertainty-aware fraud triage."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class ConformalPredictor:
    """Split conformal prediction for binary classification.

    Outputs one of: high_confidence_fraud, review_needed, high_confidence_legit.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat: float | None = None

    def fit(self, cal_scores: np.ndarray, cal_labels: np.ndarray) -> "ConformalPredictor":
        """Compute the conformal quantile from calibration scores.

        Uses nonconformity scores = |p - y| where p is calibrated probability.
        """
        nonconf = np.abs(cal_scores - cal_labels)
        n = len(nonconf)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = float(np.quantile(nonconf, q_level))
        log.info("conformal_fitted", alpha=self.alpha, q_hat=self.q_hat, n=n)
        return self

    def predict(self, calibrated_scores: np.ndarray) -> list[dict]:
        """Generate prediction sets with confidence bands."""
        assert self.q_hat is not None, "Conformal predictor not fitted"
        results = []

        for p in calibrated_scores:
            fraud_interval = (p - self.q_hat, p + self.q_hat)
            includes_fraud = fraud_interval[1] > 0.5
            includes_legit = fraud_interval[0] < 0.5

            if includes_fraud and not includes_legit:
                band = "high_confidence_fraud"
            elif includes_legit and not includes_fraud:
                band = "high_confidence_legit"
            else:
                band = "review_needed"

            results.append({
                "calibrated_score": float(p),
                "confidence_band": band,
                "interval_low": float(max(0, fraud_interval[0])),
                "interval_high": float(min(1, fraud_interval[1])),
            })

        return results

    def predict_bands(self, calibrated_scores: np.ndarray) -> np.ndarray:
        """Return just the confidence band labels as a string array."""
        preds = self.predict(calibrated_scores)
        return np.array([p["confidence_band"] for p in preds])

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.model_dir / "conformal.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ConformalPredictor":
        path = path or (cfg.model_dir / "conformal.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


def compute_conformal_metrics(
    bands: np.ndarray, labels: np.ndarray,
) -> dict:
    """Compute conformal prediction quality metrics."""
    n = len(labels)
    fraud_mask = labels == 1
    legit_mask = labels == 0

    covered_fraud = np.sum((bands == "high_confidence_fraud") & fraud_mask) + \
                    np.sum((bands == "review_needed") & fraud_mask)
    covered_legit = np.sum((bands == "high_confidence_legit") & legit_mask) + \
                    np.sum((bands == "review_needed") & legit_mask)

    coverage = float((covered_fraud + covered_legit) / n) if n > 0 else 0.0

    set_sizes = []
    for b in bands:
        if b == "review_needed":
            set_sizes.append(2)
        else:
            set_sizes.append(1)
    avg_set_size = float(np.mean(set_sizes))

    review_rate = float(np.mean(bands == "review_needed"))

    return {
        "empirical_coverage": coverage,
        "average_set_size": avg_set_size,
        "review_rate": review_rate,
        "n_high_confidence_fraud": int(np.sum(bands == "high_confidence_fraud")),
        "n_review_needed": int(np.sum(bands == "review_needed")),
        "n_high_confidence_legit": int(np.sum(bands == "high_confidence_legit")),
    }
