"""Tests for conformal prediction."""

import numpy as np
import pytest

from rift.models.conformal import conformal_triage, DecisionBand


def test_conformal_triage():
    probs = np.array([0.9, 0.1, 0.55, 0.45])
    decisions, conf = conformal_triage(probs, fraud_threshold=0.5, gap=0.2)
    assert decisions[0] == 0  # high conf fraud
    assert decisions[1] == 2   # high conf legit
    assert decisions[2] == 1   # review
    assert decisions[3] == 1   # review
