"""Deterministic replay engine for audit verification."""

from typing import Any, Optional

import duckdb

from rift.models.infer import predict
from rift.replay.recorder import init_audit_db
from rift.utils.config import AUDIT_DB_PATH


def replay(
    decision_id: str,
    model: Any,
    calibrator: Any,
    feat_cols: list[str],
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Replay a stored decision. Returns same score, band, and explanation
    when artifacts and inputs match.
    """
    con = duckdb.connect(db_path or str(AUDIT_DB_PATH))
    row = con.execute(
        "SELECT tx_payload, feat_json FROM transactions t JOIN features f ON t.decision_id = f.decision_id WHERE t.decision_id = ?",
        [decision_id],
    ).fetchone()
    if not row:
        return {"error": "decision_id not found", "decision_id": decision_id}

    tx_payload = __import__("json").loads(row[0])
    features = __import__("json").loads(row[1])

    # Re-run prediction
    pred = predict(tx_payload, model, calibrator, feat_cols)

    # Fetch original
    orig = con.execute(
        "SELECT raw_score, calibrated_score, decision_band FROM predictions WHERE decision_id = ?",
        [decision_id],
    ).fetchone()

    match = (
        orig
        and abs(pred["raw_score"] - orig[0]) < 1e-6
        and abs(pred["calibrated_score"] - orig[1]) < 1e-6
        and pred["decision_band"] == orig[2]
    )

    con.execute(
        "INSERT INTO replay_events VALUES (?, datetime('now'), ?)",
        [decision_id, 1 if match else 0],
    )
    con.close()

    return {
        "decision_id": decision_id,
        "replayed_prediction": pred,
        "original_raw_score": orig[0] if orig else None,
        "original_calibrated_score": orig[1] if orig else None,
        "original_decision_band": orig[2] if orig else None,
        "outcome_match": match,
    }
