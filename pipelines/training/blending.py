"""
Shared raw-probability blending logic.

Zindi Rule 2 compliance: this module operates ONLY on raw model
outputs. No isotonic calibration, no prior correction, no threshold
reshaping. This is the single source of truth for combining LGBM,
XGBoost, and CatBoost predictions into a submittable probability.
"""
import numpy as np


def blend_raw_probs(
    lgbm_probs: np.ndarray,
    xgb_probs: np.ndarray,
    cb_probs: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    """
    Weighted average of raw (uncalibrated) model probabilities.

    Parameters
    ----------
    lgbm_probs, xgb_probs, cb_probs : raw probability arrays, same length
    weights : (w_lgbm, w_xgb, w_cb), should sum to 1.0

    Returns
    -------
    Blended raw probabilities. No transform applied beyond the
    weighted average itself.
    """
    if not (len(lgbm_probs) == len(xgb_probs) == len(cb_probs)):
        raise ValueError(
            f"Length mismatch: lgbm={len(lgbm_probs)}, "
            f"xgb={len(xgb_probs)}, cb={len(cb_probs)}"
        )
    w_lgbm, w_xgb, w_cb = weights
    return w_lgbm * lgbm_probs + w_xgb * xgb_probs + w_cb * cb_probs
