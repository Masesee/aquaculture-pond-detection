"""
Gate tests for evaluation metrics.
Deterministic. No real data. Must pass in < 2s.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.evaluation.metrics import combined_score, evaluate


def test_combined_score_perfect():
    assert combined_score(1.0, 1.0) == pytest.approx(1.0)


def test_combined_score_zero():
    assert combined_score(0.0, 0.0) == pytest.approx(0.0)


def test_combined_score_weights():
    # 0.6*0.8 + 0.4*0.6 = 0.48 + 0.24 = 0.72
    assert combined_score(0.8, 0.6) == pytest.approx(0.72)


def test_evaluate_perfect():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs  = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result = evaluate(labels, probs)
    assert result["f1"]    == pytest.approx(1.0)
    assert result["auc"]   == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.0)


def test_evaluate_predicted_positive_rate():
    labels = np.array([0, 0, 1, 1])
    probs  = np.array([0.3, 0.4, 0.6, 0.7])
    result = evaluate(labels, probs)
    assert result["n_predicted_positive"]    == 2
    assert result["predicted_positive_rate"] == pytest.approx(0.5)


def test_evaluate_threshold_respected():
    labels = np.array([0, 1, 0, 1])
    # All probs below 0.5 → all predicted negative → F1=0
    probs = np.array([0.1, 0.4, 0.2, 0.3])
    result = evaluate(labels, probs)
    assert result["n_predicted_positive"] == 0