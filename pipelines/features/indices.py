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


def ndti(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Normalized Difference Turbidity Index.
    (red - green) / (red + green)
    Positive for turbid water (aquaculture ponds with biological load).
    Negative for clear water (reservoirs, rivers).
    Physically motivated: fish waste and algae increase red reflectance
    relative to green in shallow productive water bodies.
    """
    r = df[f"red_{month}"].astype(float)
    g = df[f"green_{month}"].astype(float)
    return (r - g) / (r + g + EPS)


def re1_nir_ratio(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Red Edge 1 to NIR ratio.
    re1 / nir
    Elevated in waters with algae/phytoplankton (aquaculture ponds).
    Suppressed in clear water and dry land.
    Values near 1.0 indicate chlorophyll fluorescence signal.
    """
    r1 = df[f"re1_{month}"].astype(float)
    n  = df[f"nir_{month}"].astype(float)
    return r1 / (n + EPS)


def swi(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Sentinel-2 Water Index.
    (re1 - swir1) / (re1 + swir1)
    Specifically designed for Sentinel-2 red-edge and SWIR bands to enhance water extraction.
    """
    re1 = df[f"re1_{month}"].astype(float)
    sw1 = df[f"swir1_{month}"].astype(float)
    return (re1 - sw1) / (re1 + sw1 + EPS)


def nfai(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Normalized Floating Algae Index.
    Measures algal/phytoplankton load in water bodies.
    NFAI = (nir - R_nir_prime) / (nir + R_nir_prime)
    Where R_nir_prime = red + 0.1873 * (swir1 - red)
    """
    red = df[f"red_{month}"].astype(float)
    nir = df[f"nir_{month}"].astype(float)
    sw1 = df[f"swir1_{month}"].astype(float)
    r_prime = red + 0.1873 * (sw1 - red)
    return (nir - r_prime) / (nir + r_prime + EPS)


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
    "NDWI":         ndwi,
    "MNDWI":        mndwi,
    "NDVI":         ndvi,
    "NDRE":         ndre,
    "AWEInsh":      awei_nsh,
    "SAR_diff_db":  sar_diff_db,
    "NDTI":         ndti,
    "re1_nir":      re1_nir_ratio,
    "SWI":          swi,
    "NFAI":         nfai,
}