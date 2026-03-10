from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalClassifier:
    alpha: float = 0.05

    def __post_init__(self) -> None:
        self.threshold: float | None = None

    def fit(self, calibrated_probabilities: np.ndarray, labels: np.ndarray) -> None:
        probs = np.asarray(calibrated_probabilities, dtype=float)
        y = np.asarray(labels, dtype=int)
        if probs.size == 0:
            self.threshold = 0.5
            return
        scores = np.where(y == 1, 1.0 - probs, probs)
        q_level = min(1.0, np.ceil((scores.size + 1) * (1.0 - self.alpha)) / scores.size)
        self.threshold = float(np.quantile(scores, q_level, method="higher"))

    def predict_sets(self, calibrated_probabilities: np.ndarray) -> list[set[str]]:
        if self.threshold is None:
            raise RuntimeError("conformal classifier is not fitted")
        prediction_sets: list[set[str]] = []
        for prob in np.asarray(calibrated_probabilities, dtype=float):
            labels = set()
            if 1.0 - prob <= self.threshold:
                labels.add("fraud")
            if prob <= self.threshold:
                labels.add("legit")
            if not labels:
                labels = {"fraud", "legit"}
            prediction_sets.append(labels)
        return prediction_sets

    def triage(self, calibrated_probabilities: np.ndarray) -> list[tuple[str, float]]:
        decisions = []
        for prob, label_set in zip(calibrated_probabilities, self.predict_sets(calibrated_probabilities), strict=False):
            if label_set == {"fraud"}:
                decisions.append(("high_confidence_fraud", float(prob)))
            elif label_set == {"legit"}:
                decisions.append(("high_confidence_legit", float(1.0 - prob)))
            else:
                decisions.append(("review_needed", float(1.0 - abs(prob - 0.5) * 2.0)))
        return decisions
