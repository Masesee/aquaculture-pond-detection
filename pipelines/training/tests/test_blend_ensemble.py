"""
Gate tests for blended ensemble pipeline functions.
Deterministic. Fast (<2s execution).
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.training.blend_ensemble import combined_score


def test_ensemble_combined_score():
    f1 = 0.85
    auc = 0.95
    score = combined_score(f1, auc)
    assert np.isclose(score, 0.6 * 0.85 + 0.4 * 0.95)


def test_blend_probabilities_math():
    p1 = np.array([0.2, 0.8, 0.6])
    p2 = np.array([0.4, 0.6, 0.8])
    blend = 0.5 * p1 + 0.5 * p2
    np.testing.assert_allclose(blend, [0.3, 0.7, 0.7])
