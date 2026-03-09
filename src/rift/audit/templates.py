"""Report templates for audit output."""

AUDIT_REPORT_TEMPLATE = """
# Rift Decision Report

**Decision ID:** {decision_id}
**Timestamp:** {timestamp}

## Summary
{summary}

## Details
{details}

## Replay
Run `rift replay {decision_id}` to verify.
"""
