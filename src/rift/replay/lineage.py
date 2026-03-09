"""Decision lineage and provenance tracking."""

from typing import Any


def lineage_from_record(
    tx_payload: dict,
    features: dict,
    prediction: dict,
    model_id: str,
    calibrator_version: str,
) -> dict[str, Any]:
    """Build lineage record for a decision."""
    return {
        "inputs": {"transaction": tx_payload, "features": features},
        "outputs": prediction,
        "artifacts": {"model_id": model_id, "calibrator_version": calibrator_version},
    }
