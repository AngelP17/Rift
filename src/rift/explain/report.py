from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import polars as pl


FRIENDLY_NAMES = {
    "amount": "transaction amount",
    "distance_from_user_centroid_km": "distance from recent user locations",
    "user_txn_count_1h": "one-hour transaction velocity",
    "user_spend_24h": "24-hour spend",
    "devices_per_account": "number of devices seen on the account",
    "device_sharing_degree": "device sharing across users",
    "merchant_fraud_prevalence": "merchant fraud prevalence",
    "seconds_since_last_txn": "time since last transaction",
    "user_amount_zscore": "amount anomaly within user history",
    "is_new_device_for_user": "new device usage",
}


def _top_drivers(feature_frame: pl.DataFrame, artifact: dict[str, Any], top_k: int = 3) -> list[tuple[str, float]]:
    columns = artifact["feature_columns"]
    values = feature_frame.select(columns).row(0)
    scores = np.abs(np.array(values, dtype=float))
    model = artifact["model"]
    if getattr(model, "model", None) is not None and hasattr(model.model, "feature_importances_"):
        weights = np.asarray(model.model.feature_importances_[: len(columns)], dtype=float)
        scores = np.abs(weights * np.asarray(values, dtype=float))
    order = np.argsort(scores)[::-1][:top_k]
    return [(columns[idx], float(scores[idx])) for idx in order]


def nearest_analogs(feature_frame: pl.DataFrame, artifact: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    ref = artifact.get("train_reference", {})
    if not ref:
        return []
    columns = artifact["feature_columns"]
    query = np.asarray(feature_frame.select(columns).row(0), dtype=float)
    references = np.asarray(ref["features"], dtype=float)
    if references.size == 0:
        return []
    distances = np.sqrt(((references - query) ** 2).sum(axis=1))
    order = np.argsort(distances)[:top_k]
    return [
        {
            "tx_id": ref["tx_ids"][idx],
            "label": int(ref["labels"][idx]),
            "distance": float(distances[idx]),
        }
        for idx in order
    ]


def counterfactual_summary(feature_frame: pl.DataFrame) -> str:
    row = feature_frame.row(0, named=True)
    actions = []
    if float(row["is_new_device_for_user"]) > 0:
        actions.append("use a previously seen device")
    if float(row["distance_from_user_centroid_km"]) > 250:
        actions.append("transact closer to the user's normal locations")
    if float(row["user_txn_count_1h"]) > 3:
        actions.append("reduce one-hour transaction velocity")
    if float(row["user_amount_zscore"]) > 2:
        actions.append("lower the transaction amount toward the user's usual range")
    if not actions:
        actions.append("provide additional context for manual review")
    return "A lower-risk alternative would be to " + ", ".join(actions[:2]) + "."


def build_explanation(feature_frame: pl.DataFrame, artifact: dict[str, Any], prediction: dict[str, Any]) -> tuple[str, list[str]]:
    drivers = _top_drivers(feature_frame, artifact)
    phrases = [FRIENDLY_NAMES.get(name, name.replace("_", " ")) for name, _ in drivers]
    explanation = (
        f"This transaction was routed to {prediction['decision']} because the strongest signals were "
        + ", ".join(phrases[:-1] + [f"and {phrases[-1]}"] if len(phrases) > 1 else phrases)
        + f". The calibrated fraud probability is {prediction['calibrated_probability']:.2%}."
    )
    return explanation, phrases


def build_audit_report(
    decision_id: str,
    feature_frame: pl.DataFrame,
    artifact: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    explanation, phrases = build_explanation(feature_frame, artifact, prediction)
    analogs = nearest_analogs(feature_frame, artifact)
    report = {
        "decision_id": decision_id,
        "decision_outcome": prediction["decision"],
        "confidence": prediction["confidence"],
        "fraud_probability": prediction["calibrated_probability"],
        "top_drivers": phrases,
        "nearest_similar_cases": analogs,
        "counterfactual_summary": counterfactual_summary(feature_frame),
        "replay_instructions": f"rift replay {decision_id}",
        "explanation": explanation,
    }
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Rift Audit Report: {report['decision_id']}",
        "",
        f"- Outcome: **{report['decision_outcome']}**",
        f"- Confidence: **{report['confidence']:.2%}**",
        f"- Calibrated fraud probability: **{report['fraud_probability']:.2%}**",
        "",
        "## Top drivers",
    ]
    lines.extend([f"- {driver}" for driver in report["top_drivers"]])
    lines.extend(
        [
            "",
            "## Explanation",
            report["explanation"],
            "",
            "## Counterfactual summary",
            report["counterfactual_summary"],
            "",
            "## Replay",
            f"`{report['replay_instructions']}`",
        ]
    )
    if report["nearest_similar_cases"]:
        lines.extend(["", "## Nearest similar prior cases"])
        for case in report["nearest_similar_cases"]:
            lines.append(
                f"- {case['tx_id']} | label={case['label']} | distance={case['distance']:.3f}"
            )
    return "\n".join(lines)
