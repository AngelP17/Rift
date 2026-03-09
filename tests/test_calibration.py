"""Tests for calibration module."""

import numpy as np

from models.calibrate import IsotonicCalibrator, PlattCalibrator, calibrate_scores


class TestPlattCalibrator:
    def test_fit_and_calibrate(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(int)

        cal = PlattCalibrator().fit(scores, labels)
        calibrated = cal.calibrate(scores)

        assert len(calibrated) == len(scores)
        assert all(0 <= c <= 1 for c in calibrated)

    def test_save_and_load(self, tmp_path):
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])

        cal = PlattCalibrator().fit(scores, labels)
        path = cal.save(tmp_path / "platt.pkl")

        loaded = PlattCalibrator.load(path)
        c1 = cal.calibrate(scores)
        c2 = loaded.calibrate(scores)
        np.testing.assert_array_almost_equal(c1, c2)


class TestIsotonicCalibrator:
    def test_fit_and_calibrate(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(int)

        cal = IsotonicCalibrator().fit(scores, labels)
        calibrated = cal.calibrate(scores)

        assert len(calibrated) == len(scores)
        assert all(0 <= c <= 1 for c in calibrated)

    def test_monotonic(self):
        scores = np.linspace(0, 1, 100)
        labels = (scores > 0.5).astype(int)

        cal = IsotonicCalibrator().fit(scores, labels)
        calibrated = cal.calibrate(scores)

        for i in range(1, len(calibrated)):
            assert calibrated[i] >= calibrated[i - 1] - 1e-10


class TestCalibrateScores:
    def test_isotonic(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        calibrated, cal = calibrate_scores(scores, labels, method="isotonic")
        assert len(calibrated) == 5

    def test_platt(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        calibrated, cal = calibrate_scores(scores, labels, method="platt")
        assert len(calibrated) == 5
