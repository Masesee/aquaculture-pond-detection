"""
Gate tests for stacking ensemble.
Deterministic. No real data. Must pass in ~ 2s.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.training.base_learners import (
    make_lgbm, make_rf, make_logreg,
    LOGREG_FEATURES,
)
from pipelines.training.ensemble import (
    _build_meta_features,
    _fit_meta_learner,
)


@pytest.fixture
def synthetic_oof() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic OOF probs + labels for 100 samples."""
    rng    = np.random.default_rng(42)
    n      = 100
    labels = rng.integers(0, 2, n)
    # Each learner has slightly different probs
    lgbm_p = np.clip(labels * 0.7 + rng.normal(0, 0.15, n), 0, 1)
    rf_p   = np.clip(labels * 0.65 + rng.normal(0, 0.18, n), 0, 1)
    lr_p   = np.clip(labels * 0.6  + rng.normal(0, 0.20, n), 0, 1)
    return lgbm_p, rf_p, lr_p, labels


def test_build_meta_features_shape(synthetic_oof):
    lgbm_p, rf_p, lr_p, _ = synthetic_oof
    meta = _build_meta_features(lgbm_p, rf_p, lr_p)
    assert meta.shape == (100, 6), f"Expected (100, 6), got {meta.shape}"


def test_build_meta_features_difference_columns(synthetic_oof):
    """Column 3 must equal col0 - col1 (lgbm - rf diff)."""
    lgbm_p, rf_p, lr_p, _ = synthetic_oof
    meta = _build_meta_features(lgbm_p, rf_p, lr_p)
    assert np.allclose(meta[:, 3], lgbm_p - rf_p)
    assert np.allclose(meta[:, 4], lgbm_p - lr_p)
    assert np.allclose(meta[:, 5], rf_p   - lr_p)


def test_fit_meta_learner_returns_objects(synthetic_oof):
    lgbm_p, rf_p, lr_p, labels = synthetic_oof
    clf, scaler = _fit_meta_learner(lgbm_p, rf_p, lr_p, labels)
    assert clf is not None
    assert scaler is not None


def test_meta_learner_proba_range(synthetic_oof):
    lgbm_p, rf_p, lr_p, labels = synthetic_oof
    clf, scaler = _fit_meta_learner(lgbm_p, rf_p, lr_p, labels)
    meta_X = _build_meta_features(lgbm_p, rf_p, lr_p)
    probs  = clf.predict_proba(scaler.transform(meta_X))[:, 1]
    assert probs.min() >= 0.0 - 1e-9
    assert probs.max() <= 1.0 + 1e-9


def test_meta_coef_shape(synthetic_oof):
    """Meta-learner has 6 input features → coef shape (1, 6)."""
    lgbm_p, rf_p, lr_p, labels = synthetic_oof
    clf, _ = _fit_meta_learner(lgbm_p, rf_p, lr_p, labels)
    assert clf.coef_.shape == (1, 6), f"Expected (1,6), got {clf.coef_.shape}"


def test_make_lgbm_params():
    """make_lgbm() must have class_weight=None and random_state=42."""
    model = make_lgbm()
    assert model.class_weight is None
    assert model.random_state == 42


def test_make_lgbm_n_estimators_override():
    model = make_lgbm(n_estimators=250)
    assert model.n_estimators == 250


def test_make_rf_reproducible():
    """Two RF models with same seed must produce identical predictions."""
    rng = np.random.default_rng(0)
    X   = pd.DataFrame(rng.standard_normal((50, 10)),
                       columns=[f"f{i}" for i in range(10)])
    y   = rng.integers(0, 2, 50)
    m1  = make_rf()
    m2  = make_rf()
    m1.fit(X, y)
    m2.fit(X, y)
    assert np.allclose(m1.predict_proba(X), m2.predict_proba(X))


def test_logreg_pipeline_has_scaler():
    """make_logreg() must return a Pipeline with a StandardScaler first step."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_logreg()
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)


def test_logreg_features_all_strings():
    assert all(isinstance(f, str) for f in LOGREG_FEATURES)
    assert len(LOGREG_FEATURES) > 0