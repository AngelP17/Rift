from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve


def pr_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(average_precision_score(labels, probabilities))


def recall_at_fpr(labels: np.ndarray, probabilities: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    valid = np.where(fpr <= target_fpr)[0]
    if valid.size == 0:
        return 0.0
    return float(tpr[valid[-1]])


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(brier_score_loss(labels, probabilities))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (probabilities >= lower) & (probabilities < upper if idx < bins - 1 else probabilities <= upper)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probabilities[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)
