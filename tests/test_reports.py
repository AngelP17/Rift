"""Tests for audit reports."""

from rift.explain.report import generate_report


def test_generate_report():
    report = generate_report(
        decision_id="abc123",
        prediction={"raw_score": 0.8, "calibrated_score": 0.75, "decision_band": "high_confidence_fraud", "confidence": 0.9},
        top_drivers_list=[("new_device", 0.3), ("geo_jump", 0.2)],
        counterfactual="Lower 'new_device' would reduce fraud score",
    )
    assert "decision_id" in report
    assert "markdown" in report
    assert "high confidence" in report["markdown"].lower() or "fraud" in report["markdown"].lower()
