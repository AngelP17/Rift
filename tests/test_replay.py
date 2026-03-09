"""Tests for replay/recorder."""

import tempfile
from pathlib import Path

import pytest

from rift.replay.hashing import canonical_hash
from rift.replay.recorder import init_audit_db, record_decision


def test_canonical_hash():
    h = canonical_hash({"a": 1, "b": 2})
    assert len(h) == 32
    assert h == canonical_hash({"b": 2, "a": 1})


def test_record_decision():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "audit.duckdb"
        con = init_audit_db(db_path)
        did = record_decision(
            con,
            {"tx_id": "t1", "amount": 100},
            {"dist": 0.5},
            {"raw_score": 0.7, "calibrated_score": 0.65, "decision_band": "review_needed"},
        )
        assert len(did) == 32
        row = con.execute("SELECT * FROM predictions WHERE decision_id = ?", [did]).fetchone()
        assert row is not None
        assert row[1] == 0.7
