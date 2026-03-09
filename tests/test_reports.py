"""Tests for report generation and audit exports."""


from audit.redact import redact_markdown, redact_report
from explain.report import generate_report, report_to_json, report_to_markdown


class TestReportGeneration:
    def test_generate_report_structure(self):
        prediction = {
            "decision_id": "DEC_TEST",
            "tx_id": "TX_001",
            "raw_score": 0.85,
            "calibrated_score": 0.82,
            "confidence_band": "high_confidence_fraud",
            "timestamp": "2024-01-15T10:30:00",
        }
        report = generate_report(prediction)
        assert "decision_id" in report
        assert "outcome" in report
        assert "confidence" in report
        assert "recommendation" in report
        assert "narrative" in report

    def test_report_to_markdown(self):
        prediction = {
            "decision_id": "DEC_TEST",
            "tx_id": "TX_001",
            "raw_score": 0.85,
            "calibrated_score": 0.82,
            "confidence_band": "high_confidence_fraud",
            "timestamp": "2024-01-15T10:30:00",
        }
        report = generate_report(prediction)
        md = report_to_markdown(report)
        assert "DEC_TEST" in md
        assert "FLAGGED AS FRAUD" in md
        assert "Recommendation" in md

    def test_report_to_json(self):
        prediction = {
            "decision_id": "DEC_TEST",
            "calibrated_score": 0.5,
            "raw_score": 0.5,
            "confidence_band": "review_needed",
            "timestamp": "2024-01-15T10:30:00",
        }
        report = generate_report(prediction)
        j = report_to_json(report)
        assert "DEC_TEST" in j

    def test_with_explanation(self):
        prediction = {
            "decision_id": "DEC_TEST",
            "calibrated_score": 0.9,
            "raw_score": 0.88,
            "confidence_band": "high_confidence_fraud",
        }
        explanation = {
            "top_features": [
                {"feature": "dist_from_centroid", "shap_value": 0.3, "feature_value": 500.0},
                {"feature": "tx_count_1h", "shap_value": 0.2, "feature_value": 15},
            ],
        }
        report = generate_report(prediction, explanation=explanation)
        assert len(report["top_drivers"]) >= 2

    def test_narrative_fraud(self):
        prediction = {
            "decision_id": "DEC_TEST",
            "calibrated_score": 0.95,
            "raw_score": 0.93,
            "confidence_band": "high_confidence_fraud",
        }
        report = generate_report(prediction)
        assert "flagged" in report["narrative"].lower()

    def test_narrative_legit(self):
        prediction = {
            "decision_id": "DEC_TEST",
            "calibrated_score": 0.05,
            "raw_score": 0.03,
            "confidence_band": "high_confidence_legit",
        }
        report = generate_report(prediction)
        assert "legitimate" in report["narrative"].lower()


class TestRedaction:
    def test_redact_report(self):
        report = {
            "decision_id": "DEC_001",
            "user_id": "U_000123",
            "device_id": "D_000456",
            "amount": 500,
        }
        redacted = redact_report(report)
        assert redacted["user_id"] == "[REDACTED]"
        assert redacted["device_id"] == "[REDACTED]"
        assert redacted["amount"] == 500

    def test_redact_markdown(self):
        md = "User U_000123 from device D_000456 made a transaction."
        redacted = redact_markdown(md)
        assert "U_000123" not in redacted
        assert "D_000456" not in redacted
        assert "[REDACTED]" in redacted
