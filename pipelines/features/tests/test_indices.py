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
    ndwi, mndwi, ndvi, ndre, awei_nsh, sar_diff_db, ndti, re1_nir_ratio,
    sabi, cdom_proxy, chlorophyll_index, ndwi2, sar_rvi,
)

MONTH = "01"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """8-row synthetic frame with realistic Sentinel-2 reflectance values."""
    rng = np.random.default_rng(0)
    n = 8
    data = {
        f"blue_{MONTH}":   rng.integers(400,  1500, n).astype(float),
        f"green_{MONTH}":  rng.integers(800,  3000, n).astype(float),
        f"red_{MONTH}":    rng.integers(600,  2500, n).astype(float),
        f"nir_{MONTH}":    rng.integers(1000, 6000, n).astype(float),
        f"nira_{MONTH}":   rng.integers(1000, 6000, n).astype(float),
        f"re1_{MONTH}":    rng.integers(800,  4000, n).astype(float),
        f"re2_{MONTH}":    rng.integers(900,  4500, n).astype(float),
        f"re3_{MONTH}":    rng.integers(900,  5000, n).astype(float),
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
        f"blue_{MONTH}":   [500.0]  * n,
        f"green_{MONTH}":  [3000.0] * n,
        f"red_{MONTH}":    [300.0]  * n,
        f"nir_{MONTH}":    [400.0]  * n,
        f"nira_{MONTH}":   [380.0]  * n,
        f"re1_{MONTH}":    [350.0]  * n,
        f"re2_{MONTH}":    [370.0]  * n,
        f"re3_{MONTH}":    [390.0]  * n,
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
        f"blue_{MONTH}":   [500.0]  * n,
        f"green_{MONTH}":  [1200.0] * n,
        f"red_{MONTH}":    [800.0]  * n,
        f"nir_{MONTH}":    [5000.0] * n,
        f"nira_{MONTH}":   [4800.0] * n,
        f"re1_{MONTH}":    [3000.0] * n,
        f"re2_{MONTH}":    [3500.0] * n,
        f"re3_{MONTH}":    [4200.0] * n,
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
    for fn in [
        ndwi, mndwi, ndvi, ndre, awei_nsh, sar_diff_db, ndti, re1_nir_ratio,
        sabi, cdom_proxy, chlorophyll_index, ndwi2, sar_rvi,
    ]:
        result = fn(sample_df, MONTH)
        assert not result.isna().any(), f"{fn.__name__} produced NaN values"


# ── New index tests (v6) ───────────────────────────────────────────────────────

def test_sabi_positive_for_algal_water():
    """
    Productive water: nir > red (algal bloom lifts NIR via fluorescence).
    SABI = (nir - red) / (blue + green) should be positive.
    """
    n = 4
    df = pd.DataFrame({
        f"nir_{MONTH}":   [4000.0] * n,   # high NIR — algal fluorescence
        f"red_{MONTH}":   [1200.0] * n,
        f"blue_{MONTH}":  [800.0]  * n,
        f"green_{MONTH}": [1000.0] * n,
    })
    vals = sabi(df, MONTH)
    assert (vals > 0).all(), f"Expected positive SABI for productive water, got {vals.values}"


def test_sabi_low_for_clear_water():
    """
    Clear water: low nir relative to red.
    SABI should be negative or near zero.
    """
    n = 4
    df = pd.DataFrame({
        f"nir_{MONTH}":   [300.0]  * n,
        f"red_{MONTH}":   [500.0]  * n,
        f"blue_{MONTH}":  [2000.0] * n,
        f"green_{MONTH}": [3000.0] * n,
    })
    vals = sabi(df, MONTH)
    assert (vals < 0).all(), f"Expected negative SABI for clear water, got {vals.values}"


def test_cdom_proxy_high_for_clear_water():
    """
    Clear water: blue > red → CDOM ratio > 1.
    """
    n = 4
    df = pd.DataFrame({
        f"blue_{MONTH}": [3000.0] * n,
        f"red_{MONTH}":  [500.0]  * n,
    })
    vals = cdom_proxy(df, MONTH)
    assert (vals > 1.0).all(), f"Expected CDOM > 1 for clear water, got {vals.values}"


def test_cdom_proxy_low_for_turbid_water():
    """
    Turbid / organic-rich water: blue < red → CDOM ratio < 1.
    """
    n = 4
    df = pd.DataFrame({
        f"blue_{MONTH}": [400.0]  * n,
        f"red_{MONTH}":  [1800.0] * n,
    })
    vals = cdom_proxy(df, MONTH)
    assert (vals < 1.0).all(), f"Expected CDOM < 1 for turbid water, got {vals.values}"


def test_cdom_proxy_positive(sample_df):
    """Ratio of two positive reflectances must always be positive."""
    vals = cdom_proxy(sample_df, MONTH)
    assert (vals > 0).all(), f"CDOM ratio must be positive, got {vals.values}"


def test_chlorophyll_index_positive_for_productive(sample_df):
    """
    In productive waters re3 > re2 (red-edge shoulder shifts to longer wavelengths).
    CI = re3/re2 - 1 should be positive when re3 > re2.
    We don't assert on sample_df since random may go either way.
    Test a known case instead.
    """
    n = 4
    df = pd.DataFrame({
        f"re3_{MONTH}": [4500.0] * n,
        f"re2_{MONTH}": [3000.0] * n,
    })
    vals = chlorophyll_index(df, MONTH)
    assert (vals > 0).all(), f"Expected CI > 0 when re3 > re2, got {vals.values}"


def test_chlorophyll_index_near_zero_equal_bands():
    """When re3 == re2, CI = 0."""
    n = 4
    df = pd.DataFrame({
        f"re3_{MONTH}": [3000.0] * n,
        f"re2_{MONTH}": [3000.0] * n,
    })
    vals = chlorophyll_index(df, MONTH)
    assert np.allclose(vals.values, 0.0, atol=1e-6), f"Expected CI~0 for equal bands, got {vals.values}"


def test_ndwi2_high_for_water():
    """
    Open water: high NIR, very low SWIR1 → NDWI2 near +1.
    """
    n = 4
    df = pd.DataFrame({
        f"nir_{MONTH}":   [5000.0] * n,
        f"swir1_{MONTH}": [100.0]  * n,
    })
    vals = ndwi2(df, MONTH)
    assert (vals > 0.9).all(), f"Expected NDWI2 > 0.9 for open water, got {vals.values}"


def test_ndwi2_low_for_bare_soil():
    """
    Dry bare soil: low NIR, high SWIR1 → NDWI2 negative.
    """
    n = 4
    df = pd.DataFrame({
        f"nir_{MONTH}":   [800.0]  * n,
        f"swir1_{MONTH}": [3500.0] * n,
    })
    vals = ndwi2(df, MONTH)
    assert (vals < 0).all(), f"Expected NDWI2 < 0 for bare soil, got {vals.values}"


def test_ndwi2_range(sample_df):
    """NDWI2 is a normalized ratio: must stay in [-1, 1]."""
    vals = ndwi2(sample_df, MONTH)
    assert vals.between(-1 - 1e-6, 1 + 1e-6).all(), f"NDWI2 out of [-1,1]: {vals.values}"


def test_sar_rvi_near_zero_for_water(pure_water_df):
    """
    Specular water: VH=-25dB, VV=-10dB.
    In linear: VH_lin << VV_lin → RVI = 4*VH_lin/(VH_lin+VV_lin) near 0.
    """
    vals = sar_rvi(pure_water_df, MONTH)
    assert (vals < 0.5).all(), f"Expected low SAR_RVI for water, got {vals.values}"


def test_sar_rvi_higher_for_vegetation(pure_veg_df):
    """
    Vegetation: VH=-14dB, VV=-10dB — more similar in linear power space than water.
    VH_lin = 10^(-1.4) ≈ 0.0398, VV_lin = 10^(-1.0) = 0.1.
    RVI = 4*0.0398/(0.0398+0.1) ≈ 1.14.
    Practically: veg RVI should be substantially higher than specular water RVI (~0.12).
    """
    veg_rvi = sar_rvi(pure_veg_df, MONTH)
    assert (veg_rvi > 0.3).all(), f"Expected SAR_RVI > 0.3 for vegetation, got {veg_rvi.values}"


def test_sar_rvi_range(sample_df):
    """SAR_RVI must be non-negative for any valid dB inputs.
    The formula 4*VH/(VH+VV) is unbounded above 1 when VH > VV in linear,
    so we only assert the lower bound.
    """
    vals = sar_rvi(sample_df, MONTH)
    assert (vals >= 0).all(), f"SAR_RVI must be non-negative, got {vals.values}"


def test_sar_rvi_known_value():
    """
    Exact arithmetic check.
    VH=-20dB → lin=0.01, VV=-10dB → lin=0.1.
    RVI = 4*0.01 / (0.01 + 0.1) = 0.04/0.11 ≈ 0.3636.
    """
    df = pd.DataFrame({
        f"VH_{MONTH}": [-20.0],
        f"VV_{MONTH}": [-10.0],
    })
    val = sar_rvi(df, MONTH).iloc[0]
    expected = (4 * 0.01) / (0.01 + 0.1)
    assert abs(val - expected) < 1e-6, f"Expected {expected:.6f}, got {val:.6f}"