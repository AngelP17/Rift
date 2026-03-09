from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str = "isotonic"

    def __post_init__(self) -> None:
        self.model: IsotonicRegression | LogisticRegression | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        x = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        if self.method == "platt":
            model = LogisticRegression(max_iter=1_000)
            model.fit(x, y)
            self.model = model
            return
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(x.ravel(), y)
        self.model = model

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        x = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        if isinstance(self.model, LogisticRegression):
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x.ravel())
