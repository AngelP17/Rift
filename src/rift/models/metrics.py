"""Evaluation metrics for fraud detection."""

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC."""
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_target: float = 0.01,
) -> float:
    """Recall at given false positive rate."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, fpr_target, side="right")
    if idx >= len(tpr):
        return 0.0
    return float(tpr[idx])


def ece(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_score[mask].mean()
        weight = mask.sum() / len(y_true)
        ece_val += weight * abs(acc - conf)
    return float(ece_val)


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score."""
    return float(brier_score_loss(y_true, y_score))


def reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean_predicted_value, mean_actual_value, bin_counts."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mean_pred = []
    mean_actual = []
    counts = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() > 0:
            mean_pred.append(y_score[mask].mean())
            mean_actual.append(y_true[mask].mean())
            counts.append(mask.sum())
        else:
            mean_pred.append((lo + hi) / 2)
            mean_actual.append(0.0)
            counts.append(0)
    return np.array(mean_pred), np.array(mean_actual), np.array(counts)
