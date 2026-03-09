"""Conformal prediction for uncertainty-aware decision bands."""

from enum import Enum
from typing import Tuple

import numpy as np


class DecisionBand(str, Enum):
    HIGH_CONFIDENCE_FRAUD = "high_confidence_fraud"
    REVIEW_NEEDED = "review_needed"
    HIGH_CONFIDENCE_LEGIT = "high_confidence_legit"


def conformal_triage(
    calibrated_probs: np.ndarray,
    alpha: float = 0.05,
    fraud_threshold: float = 0.5,
    gap: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map calibrated probabilities to decision bands.

    - high_confidence_fraud: prob > fraud_threshold + gap
    - high_confidence_legit: prob < fraud_threshold - gap
    - review_needed: otherwise

    Returns: (decisions as int 0/1/2, confidence levels)
    """
    decisions = np.ones(len(calibrated_probs), dtype=int)  # 1 = review_needed
    decisions[calibrated_probs > fraud_threshold + gap] = 0  # high_conf_fraud
    decisions[calibrated_probs < fraud_threshold - gap] = 2  # high_conf_legit

    confidence = np.abs(calibrated_probs - fraud_threshold)
    return decisions, confidence


def conformal_split(
    scores: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Conformal score threshold for target coverage (1 - alpha).
    Returns threshold such that P(true_label in prediction_set) >= 1 - alpha.
    """
    n = len(scores)
    n_cal = max(1, int(n * 0.2))
    cal_scores = scores[-n_cal:]
    cal_y = y_true[-n_cal:]

    # Nonconformity: for label 1, use 1-score; for 0, use score
    ncf = np.where(cal_y == 1, 1 - cal_scores, cal_scores)
    q = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q = min(1.0, q)
    threshold = np.quantile(ncf, q)
    return float(threshold)
