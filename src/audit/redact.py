"""PII redaction for audit reports."""

from __future__ import annotations

import re

from utils.logging import get_logger

log = get_logger(__name__)

PII_PATTERNS = {
    "user_id": r"U_\d{6}",
    "device_id": r"D_\d{6}",
    "account_id": r"A_\d{6}",
    "lat": r'"lat":\s*[\d.-]+',
    "lon": r'"lon":\s*[\d.-]+',
}

REDACT_FIELDS = {"user_id", "device_id", "account_id", "lat", "lon"}


def redact_report(report: dict, fields: set[str] | None = None) -> dict:
    """Redact PII from an audit report dict."""
    fields = fields or REDACT_FIELDS
    redacted = {}

    for key, value in report.items():
        if key in fields:
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = redact_report(value, fields)
        elif isinstance(value, list):
            redacted[key] = [
                redact_report(item, fields) if isinstance(item, dict)
                else _redact_string(str(item), fields) if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str):
            redacted[key] = _redact_string(value, fields)
        else:
            redacted[key] = value

    return redacted


def _redact_string(text: str, fields: set[str]) -> str:
    for field, pattern in PII_PATTERNS.items():
        if field in fields:
            text = re.sub(pattern, "[REDACTED]", text)
    return text


def redact_markdown(markdown: str) -> str:
    """Redact PII patterns from a markdown report."""
    for field, pattern in PII_PATTERNS.items():
        markdown = re.sub(pattern, "[REDACTED]", markdown)
    return markdown
