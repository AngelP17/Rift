"""Counterfactual explanation generator for fraud decisions."""

from __future__ import annotations

from typing import Any

import numpy as np

from explain.shap_explainer import FEATURE_NAMES
from utils.logging import get_logger

log = get_logger(__name__)


def generate_counterfactual(
    features: np.ndarray,
    predict_fn: Any,
    feature_names: list[str] | None = None,
    target_flip: bool = True,
    step_size: float = 0.1,
    max_steps: int = 50,
) -> dict:
    """Generate a counterfactual explanation.

    Finds minimal feature changes that would flip the prediction.
    Uses a greedy perturbation strategy on the most important features.
    """
    feature_names = feature_names or FEATURE_NAMES
    if features.ndim == 1:
        features = features.reshape(1, -1)

    original = features.copy()
    original_score = float(predict_fn(original)[0])
    threshold = 0.5

    target_above = original_score < threshold if target_flip else original_score >= threshold

    cf = original.copy()
    changes = []
    n_features = min(len(feature_names), cf.shape[1])

    importances = np.abs(cf[0, :n_features])
    order = np.argsort(-importances)

    for step in range(max_steps):
        current_score = float(predict_fn(cf)[0])
        if target_above and current_score >= threshold:
            break
        if not target_above and current_score < threshold:
            break

        fidx = order[step % n_features]
        direction = 1.0 if target_above else -1.0

        old_val = cf[0, fidx]
        cf[0, fidx] += direction * step_size * (abs(old_val) + 1.0)
        new_score = float(predict_fn(cf)[0])

        changes.append({
            "feature": feature_names[fidx] if fidx < len(feature_names) else f"feature_{fidx}",
            "from": float(old_val),
            "to": float(cf[0, fidx]),
            "score_change": new_score - current_score,
        })

    final_score = float(predict_fn(cf)[0])
    flipped = (final_score >= threshold) != (original_score >= threshold)

    return {
        "original_score": original_score,
        "counterfactual_score": final_score,
        "flipped": flipped,
        "changes": changes[:5],
        "summary": _summarize_counterfactual(changes[:3], flipped),
    }


def _summarize_counterfactual(changes: list[dict], flipped: bool) -> str:
    if not changes:
        return "No counterfactual changes could flip the decision."

    parts = []
    for c in changes:
        direction = "increased" if c["to"] > c["from"] else "decreased"
        parts.append(f"{c['feature']} {direction}")

    if flipped:
        return f"The decision would change if: {', '.join(parts)}."
    return f"The decision is robust. Changing {', '.join(parts)} was not sufficient to flip it."
