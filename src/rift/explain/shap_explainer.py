"""SHAP-based feature importance for explanations."""

from typing import Optional

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def explain_prediction(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    num_samples: int = 100,
) -> dict[str, float]:
    """Compute SHAP values for a single prediction (or mean over background)."""
    if not SHAP_AVAILABLE:
        return {n: 0.0 for n in feature_names}
    try:
        if hasattr(model, "predict_proba"):
            pred_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            pred_fn = model.predict_proba
        explainer = shap.KernelExplainer(pred_fn, X[:num_samples])
        shap_vals = explainer.shap_values(X[-1:], nsamples=50)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        importance = dict(zip(feature_names, shap_vals.flatten().tolist()))
        return importance
    except Exception:
        return {n: 0.0 for n in feature_names}


def top_drivers(importance: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    """Return top k drivers by absolute SHAP value."""
    sorted_ = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_[:k]
