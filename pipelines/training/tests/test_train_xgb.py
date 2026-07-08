"""
Gate tests for XGBoost training pipeline functions.
Deterministic. Fast (<2s execution).
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.training.train_xgb import combined_score, XGB_PARAMS
import xgboost as xgb


def test_combined_score_calculation():
    f1 = 0.80
    auc = 0.90
    score = combined_score(f1, auc)
    assert np.isclose(score, 0.6 * 0.80 + 0.4 * 0.90)


def test_xgb_classifier_instantiation():
    model = xgb.XGBClassifier(**XGB_PARAMS)
    assert model is not None
    assert model.get_params()["max_depth"] == 6
