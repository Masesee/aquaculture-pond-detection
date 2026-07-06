"""
Gate tests for probability calibration.
Deterministic. No real data. Must pass in ~ 2s.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.training.calibration import (
    fit_calibrator,
    apply_calibrator,
    calibration_summary,
)


@pytest.fixture
def synthetic_oof():
    """
    Synthetic OOF probabilities with known miscalibration:
    raw probs are compressed to [0.2, 0.8] while true labels span 0/1.
    A calibrator should spread them out.
    """
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, n)
    # Miscalibrated: compress probs toward 0.5
    raw_probs = np.where(labels == 1,
                         rng.uniform(0.5, 0.8, n),
                         rng.uniform(0.2, 0.5, n))
    return raw_probs, labels


def test_fit_returns_calibrator(synthetic_oof):
    probs, labels = synthetic_oof
    cal = fit_calibrator(probs, labels)
    assert cal is not None


def test_apply_output_range(synthetic_oof):
    probs, labels = synthetic_oof
    cal = fit_calibrator(probs, labels)
    calibrated = apply_calibrator(cal, probs)
    assert calibrated.min() >= 0.0 - 1e-9
    assert calibrated.max() <= 1.0 + 1e-9


def test_apply_output_shape(synthetic_oof):
    probs, labels = synthetic_oof
    cal = fit_calibrator(probs, labels)
    calibrated = apply_calibrator(cal, probs)
    assert calibrated.shape == probs.shape


def test_calibration_summary_shape(synthetic_oof):
    probs, labels = synthetic_oof
    cal = fit_calibrator(probs, labels)
    calibrated = apply_calibrator(cal, probs)
    summary = calibration_summary(probs, calibrated, labels, n_bins=10)
    assert summary.shape == (10, 7)


def test_calibrated_probs_no_nan(synthetic_oof):
    probs, labels = synthetic_oof
    cal = fit_calibrator(probs, labels)
    calibrated = apply_calibrator(cal, probs)
    assert not np.isnan(calibrated).any()


def test_mismatched_shapes_raises():
    probs  = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1])
    with pytest.raises(AssertionError):
        fit_calibrator(probs, labels)


def test_non_binary_labels_raises():
    probs  = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1, 2])
    with pytest.raises(AssertionError):
        fit_calibrator(probs, labels)


def test_prior_shift_succeeds_when_same():
    from pipelines.training.train import correct_prior
    probs = np.array([0.2, 0.5, 0.8])
    # test_prior == train_prior (no shift) should pass without errors
    corrected = correct_prior(probs, train_prior=0.4, test_prior=0.4, allow_shift=False)
    assert np.allclose(corrected, probs)


def test_prior_shift_blocked_without_flag():
    from pipelines.training.train import correct_prior
    probs = np.array([0.2, 0.5, 0.8])
    # test_prior != train_prior with allow_shift=False should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        correct_prior(probs, train_prior=0.4, test_prior=0.55, allow_shift=False)
    assert "forbidden by default" in str(excinfo.value)


def test_prior_shift_allowed_with_flag():
    from pipelines.training.train import correct_prior
    probs = np.array([0.2, 0.5, 0.8])
    # test_prior != train_prior with allow_shift=True should succeed
    corrected = correct_prior(probs, train_prior=0.4, test_prior=0.55, allow_shift=True)
    assert corrected is not None
    assert len(corrected) == len(probs)
    # The shift to 0.55 (higher positive rate) should increase probabilities
    assert np.all(corrected > probs)