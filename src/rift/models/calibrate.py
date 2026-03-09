"""Probability calibration: Isotonic and Platt scaling."""

from enum import Enum
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


class CalibrationMethod(str, Enum):
    ISOTONIC = "isotonic"
    PLATT = "platt"


def isotonic_calibrate(scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression for calibration."""
    reg = IsotonicRegression(out_of_bounds="clip")
    reg.fit(scores, y_true)
    return reg


def platt_calibrate(scores: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling (logistic regression) for calibration."""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    # Reshape for sklearn
    X = scores.reshape(-1, 1)
    lr.fit(X, y_true)
    return lr


class Calibrator:
    """Calibration layer wrapping isotonic or Platt."""

    def __init__(self, method: CalibrationMethod = CalibrationMethod.ISOTONIC):
        self.method = method
        self.model = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        if self.method == CalibrationMethod.ISOTONIC:
            self.model = isotonic_calibrate(raw_scores, y_true)
        else:
            self.model = platt_calibrate(raw_scores, y_true)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        if self.method == CalibrationMethod.ISOTONIC:
            return self.model.predict(raw_scores)
        X = raw_scores.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

    def fit_transform(self, raw_scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(raw_scores, y_true).transform(raw_scores)
