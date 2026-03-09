"""Tests for calibration."""

import numpy as np
import pytest

from rift.models.calibrate import CalibrationMethod, Calibrator


def test_isotonic_calibrator():
    np.random.seed(42)
    raw = np.random.beta(2, 5, 200)
    y = (np.random.random(200) < raw).astype(int)
    cal = Calibrator(CalibrationMethod.ISOTONIC)
    cal.fit(raw, y)
    out = cal.transform(raw[:10])
    assert len(out) == 10
    assert np.all((out >= 0) & (out <= 1))
