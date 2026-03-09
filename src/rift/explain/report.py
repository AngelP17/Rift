"""Plain-English audit report generator."""

from datetime import datetime, timezone
from typing import Any, Optional

from rift.explain.counterfactuals import counterfactual_summary
from rift.explain.shap_explainer import top_drivers


def generate_report(
    decision_id: str,
    prediction: dict[str, Any],
    top_drivers_list: list[tuple[str, float]],
    counterfactual: str,
    nearest_cases: Optional[list[tuple[int, float, int]]] = None,
    tx_summary: Optional[str] = None,
) -> dict[str, Any]:
    """Generate structured audit report (markdown + JSON)."""

    band = prediction.get("decision_band", "review_needed")
    if "high_confidence_fraud" in band:
        outcome = "Flagged as fraud (high confidence)"
        review_rec = "Recommend blocking and manual investigation."
    elif "high_confidence_legit" in band:
        outcome = "Cleared as legitimate (high confidence)"
        review_rec = "No review needed."
    else:
        outcome = "Uncertain - review needed"
        review_rec = "Recommend manual review before approving."

    drivers_text = ", ".join([f"{n} ({v:.2f})" for n, v in top_drivers_list[:5]]) or "N/A"
    nearest_text = ""
    if nearest_cases:
        fraud_count = sum(1 for _, _, l in nearest_cases if l == 1)
        nearest_text = f"Of the 5 most similar prior cases, {fraud_count} were fraudulent."

    md = f"""# Rift Audit Report

**Decision ID:** {decision_id}
**Generated:** {datetime.now(timezone.utc).isoformat()}Z

## Decision
{outcome}

**Raw Score:** {prediction.get('raw_score', 0):.4f}
**Calibrated Score:** {prediction.get('calibrated_score', 0):.4f}
**Confidence Level:** {prediction.get('confidence', 0):.4f}

## Top Factors
{drivers_text}

## Counterfactual
{counterfactual}

## Similar Cases
{nearest_text}

## Reviewer Recommendation
{review_rec}

## Replay
To verify this decision: `rift replay {decision_id}`
"""
    return {
        "decision_id": decision_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_outcome": outcome,
        "confidence_level": prediction.get("confidence"),
        "top_drivers": [{"name": n, "value": v} for n, v in top_drivers_list],
        "counterfactual_summary": counterfactual,
        "nearest_cases": nearest_cases,
        "reviewer_recommendation": review_rec,
        "markdown": md,
    }
