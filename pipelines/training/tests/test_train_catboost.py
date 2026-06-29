"""
Gate tests for CatBoost training pipeline functions.
Deterministic. Fast (<2s execution).
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.training.train_catboost import correct_prior, combined_score, CAT_PARAMS
import catboost as cb


def test_correct_prior_bounds():
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    corrected = correct_prior(probs, train_prior=0.404, test_prior=0.550)
    assert len(corrected) == len(probs)
    assert (corrected >= 0.0).all() and (corrected <= 1.0).all()


def test_combined_score_calculation():
    f1 = 0.80
    auc = 0.90
    score = combined_score(f1, auc)
    assert np.isclose(score, 0.6 * 0.80 + 0.4 * 0.90)


def test_cat_classifier_instantiation():
    model = cb.CatBoostClassifier(**CAT_PARAMS)
    assert model is not None
    assert model.get_params()["depth"] == 6
