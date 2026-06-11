"""
Gate tests for index computation.
Deterministic. No real data. Must pass in < 2s.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.features.indices import (
    ndwi, mndwi, ndvi, ndre, awei_nsh, sar_diff_db, ndti, re1_nir_ratio
)

MONTH = "01"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """8-row synthetic frame with realistic Sentinel-2 reflectance values."""
    rng = np.random.default_rng(0)
    n = 8
    data = {
        f"green_{MONTH}":  rng.integers(800,  3000, n).astype(float),
        f"red_{MONTH}":    rng.integers(600,  2500, n).astype(float),
        f"nir_{MONTH}":    rng.integers(1000, 6000, n).astype(float),
        f"nira_{MONTH}":   rng.integers(1000, 6000, n).astype(float),
        f"re1_{MONTH}":    rng.integers(800,  4000, n).astype(float),
        f"swir1_{MONTH}":  rng.integers(500,  4000, n).astype(float),
        f"swir2_{MONTH}":  rng.integers(300,  3000, n).astype(float),
        f"VH_{MONTH}":     rng.uniform(-28, -10, n),
        f"VV_{MONTH}":     rng.uniform(-18,  -5, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def pure_water_df() -> pd.DataFrame:
    """
    Simulated pure water pixel:
    - high green, low nir, low red, low swir → NDWI/MNDWI strongly positive
    - VH << VV in dB → sar_diff_db very negative
    """
    n = 4
    return pd.DataFrame({
        f"green_{MONTH}":  [3000.0] * n,
        f"red_{MONTH}":    [300.0]  * n,
        f"nir_{MONTH}":    [400.0]  * n,
        f"nira_{MONTH}":   [380.0]  * n,
        f"re1_{MONTH}":    [350.0]  * n,
        f"swir1_{MONTH}":  [200.0]  * n,
        f"swir2_{MONTH}":  [150.0]  * n,
        f"VH_{MONTH}":     [-25.0]  * n,
        f"VV_{MONTH}":     [-10.0]  * n,
    })


@pytest.fixture
def pure_veg_df() -> pd.DataFrame:
    """
    Simulated dense vegetation:
    - high nir, moderate red → NDVI strongly positive
    - low green relative to nir → NDWI negative
    """
    n = 4
    return pd.DataFrame({
        f"green_{MONTH}":  [1200.0] * n,
        f"red_{MONTH}":    [800.0]  * n,
        f"nir_{MONTH}":    [5000.0] * n,
        f"nira_{MONTH}":   [4800.0] * n,
        f"re1_{MONTH}":    [3000.0] * n,
        f"swir1_{MONTH}":  [1500.0] * n,
        f"swir2_{MONTH}":  [900.0]  * n,
        f"VH_{MONTH}":     [-14.0]  * n,
        f"VV_{MONTH}":     [-10.0]  * n,
    })


# ── Range tests ────────────────────────────────────────────────────────────────

def test_ndwi_range(sample_df):
    vals = ndwi(sample_df, MONTH)
    assert vals.between(-1 - 1e-6, 1 + 1e-6).all(), f"NDWI out of [-1,1]: {vals.values}"


def test_mndwi_range(sample_df):
    vals = mndwi(sample_df, MONTH)
    assert vals.between(-1 - 1e-6, 1 + 1e-6).all()


def test_ndvi_range(sample_df):
    vals = ndvi(sample_df, MONTH)
    assert vals.between(-1 - 1e-6, 1 + 1e-6).all()


def test_ndre_range(sample_df):
    vals = ndre(sample_df, MONTH)
    assert vals.between(-1 - 1e-6, 1 + 1e-6).all()


def test_ndti_positive_for_turbid_water():
    """
    Turbid water: red > green → NDTI positive.
    Aquaculture ponds with biological load have elevated red reflectance.
    """
    n = 4
    df = pd.DataFrame({
        f"red_{MONTH}":   [1200.0] * n,   # elevated red — turbidity
        f"green_{MONTH}": [900.0]  * n,   # lower green
    })
    vals = ndti(df, MONTH)
    assert (vals > 0).all(), f"Expected positive NDTI for turbid water, got {vals.values}"


def test_ndti_negative_for_clear_water():
    """
    Clear water: green > red → NDTI negative.
    Rivers and reservoirs scatter more in green than red.
    """
    n = 4
    df = pd.DataFrame({
        f"red_{MONTH}":   [300.0]  * n,
        f"green_{MONTH}": [3000.0] * n,
    })
    vals = ndti(df, MONTH)
    assert (vals < 0).all(), f"Expected negative NDTI for clear water, got {vals.values}"


def test_ndti_range(sample_df):
    vals = ndti(sample_df, MONTH)
    assert vals.between(-1 - 1e-6, 1 + 1e-6).all()


def test_re1_nir_ratio_positive(sample_df):
    vals = re1_nir_ratio(sample_df, MONTH)
    assert (vals > 0).all()


def test_re1_nir_ratio_near_one_for_algae():
    """
    When re1 ≈ nir (algae fluorescence), ratio approaches 1.0.
    """
    n = 4
    df = pd.DataFrame({
        f"re1_{MONTH}": [3000.0] * n,
        f"nir_{MONTH}": [3100.0] * n,
    })
    vals = re1_nir_ratio(df, MONTH)
    assert (vals > 0.9).all(), f"Expected ratio near 1.0 for algae signal, got {vals.values}"


def test_sar_diff_db_is_negative_for_water(pure_water_df):
    """VH=-25, VV=-10 → diff = -15. Water = negative SAR diff."""
    vals = sar_diff_db(pure_water_df, MONTH)
    assert (vals < 0).all(), f"Expected negative SAR diff for water, got {vals.values}"
    assert np.allclose(vals.values, -15.0), f"Expected -15.0, got {vals.values}"


def test_ndwi_positive_for_water(pure_water_df):
    vals = ndwi(pure_water_df, MONTH)
    assert (vals > 0).all(), f"Expected positive NDWI for water, got {vals.values}"


def test_ndwi_negative_for_vegetation(pure_veg_df):
    vals = ndwi(pure_veg_df, MONTH)
    assert (vals < 0).all(), f"Expected negative NDWI for vegetation, got {vals.values}"


def test_ndvi_positive_for_vegetation(pure_veg_df):
    vals = ndvi(pure_veg_df, MONTH)
    assert (vals > 0).all(), f"Expected positive NDVI for vegetation, got {vals.values}"


def test_ndvi_near_zero_for_water(pure_water_df):
    """Water: nir=400, red=300 → NDVI = (400-300)/(400+300) ≈ 0.143. Low but positive."""
    vals = ndvi(pure_water_df, MONTH)
    assert (vals < 0.25).all(), f"Expected low NDVI for water, got {vals.values}"


def test_awei_nsh_positive_for_water(pure_water_df):
    vals = awei_nsh(pure_water_df, MONTH)
    assert (vals > 0).all(), f"Expected positive AWEInsh for water, got {vals.values}"


def test_no_nan_outputs(sample_df):
    """No index should produce NaN for valid inputs."""
    for fn in [ndwi, mndwi, ndvi, ndre, awei_nsh, sar_diff_db, ndti, re1_nir_ratio]:
        result = fn(sample_df, MONTH)
        assert not result.isna().any(), f"{fn.__name__} produced NaN values"