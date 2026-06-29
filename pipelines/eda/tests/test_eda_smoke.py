"""
Gate tests for EDA pipeline.
These are deterministic, run locally, must pass in < 2s.
They do NOT require the real data — they use minimal synthetic fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from contracts.schema import MONTHS, ALL_BANDS, TARGET_COL, raw_col
from pipelines.eda.spectral_separation import _ndwi, _sar_ratio


@pytest.fixture
def minimal_train() -> pd.DataFrame:
    """8-row synthetic train dataframe with all required columns."""
    rng = np.random.default_rng(42)
    n = 8
    data = {"ID": [f"ID_TR_{i:04d}" for i in range(n)]}
    data[TARGET_COL] = [0, 0, 0, 0, 1, 1, 1, 1]

    for band in ALL_BANDS:
        for month in MONTHS:
            data[raw_col(band, month)] = rng.uniform(500, 6000, n)

    # Give SAR bands realistic negative dB values
    for month in MONTHS:
        data[f"VH_{month}"] = rng.uniform(-30, -10, n)
        data[f"VV_{month}"] = rng.uniform(-20, -5, n)

    return pd.DataFrame(data)


def test_ndwi_range(minimal_train):
    """NDWI must be in [-1, 1] for all rows and months."""
    for month in MONTHS:
        vals = _ndwi(minimal_train, month)
        assert vals.between(-1 - 1e-6, 1 + 1e-6).all(), (
            f"NDWI out of range for month {month}: min={vals.min():.4f} max={vals.max():.4f}"
        )


def test_sar_ratio_positive(minimal_train):
    """SAR ratio (linear VH/VV) must be positive."""
    for month in MONTHS:
        vals = _sar_ratio(minimal_train, month)
        assert (vals > 0).all(), f"SAR ratio has non-positive values in month {month}"


def test_schema_validate_train(minimal_train):
    """DataSchema.validate_train must pass on a well-formed fixture."""
    from contracts.schema import DataSchema
    DataSchema.validate_train(minimal_train)  # should not raise


def test_schema_rejects_missing_column(minimal_train):
    """DataSchema must raise when a required column is dropped."""
    from contracts.schema import DataSchema
    broken = minimal_train.drop(columns=["VH_01"])
    with pytest.raises(ValueError, match="Missing columns"):
        DataSchema.validate_train(broken)