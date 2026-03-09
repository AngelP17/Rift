"""Data redaction for audit/privacy compliance."""

from typing import Any


# Fields to redact in audit exports
REDACT_FIELDS = {"user_id", "device_id", "account_id", "lat", "lon", "email", "phone"}


def redact_payload(obj: Any, fields: set[str] | None = None) -> Any:
    """Redact sensitive fields from payload."""
    fields = fields or REDACT_FIELDS
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if k.lower() in fields else redact_payload(v, fields)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_payload(v, fields) for v in obj]
    return obj
