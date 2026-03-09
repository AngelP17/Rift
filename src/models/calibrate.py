"""Probability calibration: Platt scaling and isotonic regression."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class PlattCalibrator:
    """Platt scaling: logistic regression on raw scores."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        self.model.fit(scores.reshape(-1, 1), labels)
        log.info("platt_calibrator_fitted", n=len(labels))
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.model_dir / "calibrator_platt.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "PlattCalibrator":
        path = path or (cfg.model_dir / "calibrator_platt.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


class IsotonicCalibrator:
    """Isotonic regression calibration for monotone probability mapping."""

    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(scores, labels)
        log.info("isotonic_calibrator_fitted", n=len(labels))
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict(scores)

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.model_dir / "calibrator_isotonic.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "IsotonicCalibrator":
        path = path or (cfg.model_dir / "calibrator_isotonic.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


def calibrate_scores(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    method: str = "isotonic",
) -> tuple[np.ndarray, PlattCalibrator | IsotonicCalibrator]:
    """Fit a calibrator and return calibrated scores + the calibrator object."""
    if method == "platt":
        cal = PlattCalibrator().fit(raw_scores, labels)
    elif method == "isotonic":
        cal = IsotonicCalibrator().fit(raw_scores, labels)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibrated = cal.calibrate(raw_scores)
    return calibrated, cal
