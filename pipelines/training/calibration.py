"""
Probability calibration via isotonic regression.

Fit on out-of-fold (OOF) predictions from the main model.
Applied to test set probabilities before submission.

Why isotonic over Platt (sigmoid):
  - Isotonic is non-parametric: no assumption about the shape of miscalibration
  - LightGBM miscalibration is rarely purely sigmoid-shaped
  - With ~963 OOF samples, isotonic has enough data to fit without overfitting

Why calibrate at all given near-balanced classes:
  - 40.8% positive rate means raw model probabilities may still be offset
  - F1 at fixed 0.5 threshold is 60% of the score — a 5% shift in the
    probability distribution's centre of mass changes ~50 predictions
  - One-time cost, zero downside if already well-calibrated
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import joblib
from pathlib import Path


def fit_calibrator(
    oof_probs: np.ndarray,
    labels: np.ndarray,
) -> IsotonicRegression:
    """
    Fits an isotonic regression calibrator on OOF probabilities.

    Parameters
    ----------
    oof_probs : (n,) float array of raw model probabilities from OOF predictions
    labels    : (n,) int array of true binary labels

    Returns
    -------
    Fitted IsotonicRegression instance.
    """
    assert oof_probs.shape == labels.shape, "oof_probs and labels must have the same shape"
    assert set(np.unique(labels)).issubset({0, 1}), "labels must be binary"

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, labels)
    return calibrator


def apply_calibrator(
    calibrator: IsotonicRegression,
    raw_probs: np.ndarray,
) -> np.ndarray:
    """
    Applies a fitted calibrator to raw model probabilities.

    Parameters
    ----------
    calibrator : fitted IsotonicRegression
    raw_probs  : (n,) float array

    Returns
    -------
    (n,) float array of calibrated probabilities, clipped to [0, 1].
    """
    calibrated = calibrator.predict(raw_probs)
    return np.clip(calibrated, 0.0, 1.0)


def calibration_summary(
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Produces a bin-level calibration summary for inspection.
    Each bin shows mean predicted probability vs actual positive rate.

    A well-calibrated model has predicted ≈ actual across all bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask_raw  = (raw_probs  >= lo) & (raw_probs  < hi)
        mask_cal  = (calibrated_probs >= lo) & (calibrated_probs < hi)
        rows.append({
            "bin":              f"[{lo:.1f},{hi:.1f})",
            "n_raw":            mask_raw.sum(),
            "mean_pred_raw":    raw_probs[mask_raw].mean()       if mask_raw.sum() > 0 else np.nan,
            "actual_rate_raw":  labels[mask_raw].mean()          if mask_raw.sum() > 0 else np.nan,
            "n_cal":            mask_cal.sum(),
            "mean_pred_cal":    calibrated_probs[mask_cal].mean() if mask_cal.sum() > 0 else np.nan,
            "actual_rate_cal":  labels[mask_cal].mean()           if mask_cal.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_calibrator(calibrator: IsotonicRegression, path: Path) -> None:
    joblib.dump(calibrator, path)


def load_calibrator(path: Path) -> IsotonicRegression:
    return joblib.load(path)