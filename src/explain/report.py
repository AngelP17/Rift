"""Plain-English audit report generator."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from utils.logging import get_logger

log = get_logger(__name__)

CONFIDENCE_LABELS = {
    "high_confidence_fraud": "High",
    "review_needed": "Medium (review recommended)",
    "high_confidence_legit": "Low",
}

RECOMMENDATION_MAP = {
    "high_confidence_fraud": "Block transaction and escalate to fraud investigation team.",
    "review_needed": "Route to manual review queue for analyst investigation.",
    "high_confidence_legit": "Approve transaction. No further action required.",
}


def generate_report(
    prediction: dict,
    explanation: dict | None = None,
    counterfactual: dict | None = None,
    nearest_cases: list[dict] | None = None,
) -> dict:
    """Generate a structured audit report for a prediction."""
    band = prediction.get("confidence_band", "review_needed")
    decision_id = prediction.get("decision_id", "UNKNOWN")

    top_drivers = []
    if explanation and "top_features" in explanation:
        for feat in explanation["top_features"][:5]:
            top_drivers.append(_describe_feature(feat))

    case_descriptions = []
    if nearest_cases:
        for case in nearest_cases[:3]:
            label = "fraudulent" if case.get("is_fraud") else "legitimate"
            case_descriptions.append(
                f"Transaction {case.get('tx_id', 'unknown')} ({label}, "
                f"similarity: {case.get('similarity', 0):.2f})"
            )

    cf_summary = ""
    if counterfactual:
        cf_summary = counterfactual.get("summary", "")

    narrative = _build_narrative(prediction, top_drivers, band)

    report = {
        "decision_id": decision_id,
        "decision_time": prediction.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "outcome": _outcome_text(band),
        "confidence": CONFIDENCE_LABELS.get(band, "Unknown"),
        "calibrated_score": prediction.get("calibrated_score", 0),
        "raw_score": prediction.get("raw_score", 0),
        "top_drivers": top_drivers,
        "nearest_cases": case_descriptions,
        "counterfactual": cf_summary,
        "replay_instructions": f"Run: rift replay {decision_id}",
        "recommendation": RECOMMENDATION_MAP.get(band, "Review needed."),
        "narrative": narrative,
    }

    return report


def report_to_markdown(report: dict) -> str:
    """Convert a structured report to markdown format."""
    lines = [
        f"# Audit Report: {report['decision_id']}",
        "",
        f"**Decision Time:** {report['decision_time']}",
        f"**Outcome:** {report['outcome']}",
        f"**Confidence Level:** {report['confidence']}",
        f"**Calibrated Score:** {report['calibrated_score']:.4f}",
        f"**Raw Score:** {report['raw_score']:.4f}",
        "",
        "## Summary",
        "",
        report.get("narrative", ""),
        "",
        "## Top Risk Drivers",
        "",
    ]

    for i, driver in enumerate(report.get("top_drivers", []), 1):
        lines.append(f"{i}. {driver}")

    lines.extend(["", "## Similar Historical Cases", ""])
    for case in report.get("nearest_cases", []):
        lines.append(f"- {case}")
    if not report.get("nearest_cases"):
        lines.append("- No similar cases found.")

    lines.extend(["", "## Counterfactual Analysis", ""])
    lines.append(report.get("counterfactual", "Not available.") or "Not available.")

    lines.extend([
        "",
        "## Recommendation",
        "",
        report.get("recommendation", ""),
        "",
        "## Replay",
        "",
        report.get("replay_instructions", ""),
        "",
        "---",
        f"*Report generated at {datetime.now(timezone.utc).isoformat()}*",
    ])

    return "\n".join(lines)


def report_to_json(report: dict) -> str:
    """Convert a structured report to JSON string."""
    return json.dumps(report, indent=2, default=str)


def _outcome_text(band: str) -> str:
    mapping = {
        "high_confidence_fraud": "FLAGGED AS FRAUD",
        "review_needed": "SENT TO REVIEW",
        "high_confidence_legit": "APPROVED",
    }
    return mapping.get(band, "UNKNOWN")


def _describe_feature(feat: dict) -> str:
    """Convert a feature explanation into plain English."""
    name = feat.get("feature", "unknown")
    value = feat.get("feature_value", 0)
    shap = feat.get("shap_value", 0)

    descriptions = {
        "dist_from_centroid": f"Location was {value:.1f} km from the user's usual area",
        "tx_count_1h": f"User made {value:.0f} transactions in the last hour",
        "tx_count_24h": f"User made {value:.0f} transactions in the last 24 hours",
        "spend_1h": f"User spent ${value:.2f} in the last hour",
        "spend_24h": f"User spent ${value:.2f} in the last 24 hours",
        "amount": f"Transaction amount was ${value:.2f}",
        "amount_zscore": f"Amount was {abs(value):.1f} standard deviations {'above' if value > 0 else 'below'} user average",
        "device_sharing_degree": f"Device is shared by {value:.0f} users",
        "devices_per_account": f"Account uses {value:.0f} different devices",
        "merchant_fraud_rate": f"Merchant has a {value * 100:.1f}% historical fraud rate",
        "time_since_last_tx": f"Last transaction was {value:.0f} seconds ago",
        "new_merchants_24h": f"User visited {value:.0f} new merchants in 24 hours",
        "channel_web": "Transaction was via web channel" if value > 0 else "Transaction was not via web",
    }

    desc = descriptions.get(name, f"{name} = {value:.4f}")
    direction = "(increased risk)" if shap > 0 else "(decreased risk)"
    return f"{desc} {direction}"


def _build_narrative(prediction: dict, drivers: list[str], band: str) -> str:
    score = prediction.get("calibrated_score", 0)

    if band == "high_confidence_fraud":
        opener = "This transaction was flagged as likely fraudulent."
    elif band == "review_needed":
        opener = "This transaction requires manual review due to uncertain risk signals."
    else:
        opener = "This transaction appears legitimate."

    driver_text = ""
    if drivers:
        driver_text = " Key factors: " + "; ".join(drivers[:3]) + "."

    confidence_text = f" The calibrated risk score is {score:.2%}."

    return f"{opener}{confidence_text}{driver_text}"
