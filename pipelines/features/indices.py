"""
Spectral and SAR index computation — per month.
All functions take a DataFrame and a month string, return a Series.
No state. No side effects. Pure functions only.

Design note: optical bands are in surface reflectance integer units
(Sentinel-2 L2A, scaled ×10000). SAR bands are in dB (negative floats).
"""

import pandas as pd

EPS = 1e-9  # prevents division by zero; negligible vs reflectance values


# ── Optical indices ────────────────────────────────────────────────────────────

def ndwi(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Normalized Difference Water Index.
    (green - nir) / (green + nir)
    Positive → water. Negative → vegetation / dry land.
    """
    g = df[f"green_{month}"].astype(float)
    n = df[f"nir_{month}"].astype(float)
    return (g - n) / (g + n + EPS)


def mndwi(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Modified NDWI. (green - swir1) / (green + swir1)
    Better than NDWI at suppressing built-up false positives.
    """
    g  = df[f"green_{month}"].astype(float)
    s1 = df[f"swir1_{month}"].astype(float)
    return (g - s1) / (g + s1 + EPS)


def ndvi(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Normalized Difference Vegetation Index.
    (nir - red) / (nir + red)
    Near-zero or negative for water. High for dense vegetation.
    """
    n = df[f"nir_{month}"].astype(float)
    r = df[f"red_{month}"].astype(float)
    return (n - r) / (n + r + EPS)


def ndre(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Normalized Difference Red Edge.
    (nira - re1) / (nira + re1)
    Low and stable for water. Rises for photosynthetically active vegetation.
    """
    na = df[f"nira_{month}"].astype(float)
    r1 = df[f"re1_{month}"].astype(float)
    return (na - r1) / (na + r1 + EPS)


def awei_nsh(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Automated Water Extraction Index (no shadow).
    4*(green - swir1) - (0.25*nir + 2.75*swir2)
    Strongly positive for open water. Designed for Sentinel-2 scale.
    Divide by 10000 to keep values in a sane range for tree models.
    """
    g  = df[f"green_{month}"].astype(float)
    n  = df[f"nir_{month}"].astype(float)
    s1 = df[f"swir1_{month}"].astype(float)
    s2 = df[f"swir2_{month}"].astype(float)
    raw = 4 * (g - s1) - (0.25 * n + 2.75 * s2)
    return raw / 10_000.0


# ── SAR indices ────────────────────────────────────────────────────────────────

def sar_diff_db(df: pd.DataFrame, month: str) -> pd.Series:
    """
    VH - VV in dB space.
    Water: very negative (VH much lower than VV due to specular reflection).
    Rough surfaces / vegetation: less negative.
    Both bands already in dB, so subtraction is correct.
    """
    vh = df[f"VH_{month}"].astype(float)
    vv = df[f"VV_{month}"].astype(float)
    return vh - vv


# ── Index registry ─────────────────────────────────────────────────────────────
# Maps index name → function. Consumed by aggregations.py.

INDEX_FN_MAP: dict[str, callable] = {
    "NDWI":       ndwi,
    "MNDWI":      mndwi,
    "NDVI":       ndvi,
    "NDRE":       ndre,
    "AWEInsh":    awei_nsh,
    "SAR_diff_db": sar_diff_db,
}