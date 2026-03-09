"""SHAP-based feature importance explainer for fraud decisions."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)

FEATURE_NAMES = [
    "tx_count_1h", "tx_count_24h", "tx_count_7d",
    "spend_1h", "spend_24h", "spend_7d",
    "dist_from_centroid",
    "new_merchants_24h", "new_merchants_7d",
    "devices_per_account",
    "merchant_fraud_rate",
    "device_sharing_degree",
    "time_since_last_tx",
    "amount_zscore",
    "amount", "lat", "lon",
    "channel_web", "channel_mobile", "channel_pos",
]


class ShapExplainer:
    """Compute SHAP values for XGBoost-based models."""

    def __init__(self, model: Any, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            if hasattr(self.model, "model") and self.model.model is not None:
                self._explainer = shap.TreeExplainer(self.model.model)
            else:
                self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain(self, features: np.ndarray) -> dict:
        """Generate SHAP explanation for a single prediction."""
        try:
            explainer = self._get_explainer()
            if features.ndim == 1:
                features = features.reshape(1, -1)

            shap_values = explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            values = shap_values[0] if shap_values.ndim > 1 else shap_values
            names = self.feature_names[:len(values)]

            ranked = sorted(
                zip(names, values.tolist(), features[0].tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            return {
                "shap_values": dict(zip(names, values.tolist())),
                "top_features": [
                    {"feature": name, "shap_value": sv, "feature_value": fv}
                    for name, sv, fv in ranked[:5]
                ],
                "base_value": float(explainer.expected_value if isinstance(explainer.expected_value, float) else explainer.expected_value[0]),
            }
        except Exception as e:
            log.warning("shap_fallback", error=str(e))
            return _heuristic_explanation(features, self.feature_names)


def _heuristic_explanation(features: np.ndarray, names: list[str]) -> dict:
    """Fallback heuristic explanation when SHAP is unavailable."""
    if features.ndim == 1:
        features = features.reshape(1, -1)

    values = features[0]
    n = min(len(names), len(values))

    ranked = sorted(
        [(names[i], float(values[i])) for i in range(n)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "shap_values": {},
        "top_features": [
            {"feature": name, "shap_value": 0.0, "feature_value": val}
            for name, val in ranked[:5]
        ],
        "base_value": 0.0,
        "note": "heuristic_fallback",
    }
