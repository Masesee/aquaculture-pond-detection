import numpy as np
import pytest
from pipelines.training.blending import blend_raw_probs


def test_blend_is_weighted_average():
    lgbm = np.array([0.1, 0.9])
    xgb = np.array([0.2, 0.8])
    cb = np.array([0.3, 0.7])
    result = blend_raw_probs(lgbm, xgb, cb, weights=(0.5, 0.3, 0.2))
    expected = 0.5 * lgbm + 0.3 * xgb + 0.2 * cb
    np.testing.assert_array_almost_equal(result, expected)


def test_blend_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        blend_raw_probs(
            np.array([0.1, 0.2]), np.array([0.1]), np.array([0.1, 0.2]),
            weights=(0.34, 0.33, 0.33),
        )


def test_blend_output_is_raw_no_reshaping():
    """Regression gate: output must be a pure linear combination of
    inputs — no monotonic remap, no clipping beyond what the weights
    naturally produce. Protects Zindi Rule 2 compliance."""
    lgbm = np.linspace(0, 1, 50)
    xgb = np.linspace(0, 1, 50)
    cb = np.linspace(0, 1, 50)
    result = blend_raw_probs(lgbm, xgb, cb, weights=(1 / 3, 1 / 3, 1 / 3))
    np.testing.assert_array_almost_equal(result, lgbm)  # all inputs equal -> output equals input exactly
