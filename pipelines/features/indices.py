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


def sabi(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Surface Algal Bloom Index.
    (nir - red) / (blue + green)
    Targets algal bloom signature in productive water bodies.
    Aquaculture ponds with dense phytoplankton → elevated SABI.
    Clear reservoirs / rivers → low SABI.
    Different normalization from NDVI (blue+green denominator suppresses
    turbid-water scattering that NDVI conflates with vegetation).
    Reference: Alawadi 2010.
    """
    n  = df[f"nir_{month}"].astype(float)
    r  = df[f"red_{month}"].astype(float)
    b  = df[f"blue_{month}"].astype(float)
    g  = df[f"green_{month}"].astype(float)
    return (n - r) / (b + g + EPS)


def cdom_proxy(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Colored Dissolved Organic Matter proxy.
    blue / red
    Clear water: high blue, low red → high ratio (>1).
    Turbid / organic-rich water (aquaculture, humic lakes): low ratio (<1).
    Orthogonal to NDTI which uses (red-green)/(red+green).
    This ratio is unbounded above, so tree models handle it naturally.
    """
    b = df[f"blue_{month}"].astype(float)
    r = df[f"red_{month}"].astype(float)
    return b / (r + EPS)


def chlorophyll_index(df: pd.DataFrame, month: str) -> pd.Series:
    """
    Red-Edge Chlorophyll Index (Gitelson et al.).
    re3 / re2 - 1
    Tracks chlorophyll-a fluorescence via the red-edge inflection point.
    Peaks for waters with dense phytoplankton (aquaculture production season).
    Low and stable for clear water and dry land.
    Uses B7 (re3, ~783 nm) and B6 (re2, ~740 nm).
    """
    r3 = df[f"re3_{month}"].astype(float)
    r2 = df[f"re2_{month}"].astype(float)
    return (r3 / (r2 + EPS)) - 1.0


def ndwi2(df: pd.DataFrame, month: str) -> pd.Series:
    """
    NIR–SWIR1 Normalized Difference Water Index (Gao 1996).
    (nir - swir1) / (nir + swir1)
    Sensitive to liquid water content in vegetation canopy and soil.
    For open water: near 1.0. Dry bare soil: near -1.0.
    Orthogonal to McFeeters NDWI (green-nir) because the NIR fluorescence
    bump from algae shifts nir differently than green.
    """
    n  = df[f"nir_{month}"].astype(float)
    s1 = df[f"swir1_{month}"].astype(float)
    return (n - s1) / (n + s1 + EPS)


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


def sar_rvi(df: pd.DataFrame, month: str) -> pd.Series:
    """
    SAR Radar Vegetation Index (Kim & van Zyl 2009).
    Converts VH and VV from dB to linear power before computing:
        RVI = (4 * VH_linear) / (VH_linear + VV_linear)
    Range: [0, 1].
    Open specular water: VH << VV in linear → RVI near 0.
    Dense vegetation / rough surface: VH approaches VV → RVI near 1.
    Independent from SAR_diff_db: that index stays in dB log-space;
    RVI encodes the ratio in linear power space — a different functional
    form that may separate edge cases where log-space diff is ambiguous.
    """
    vh_db = df[f"VH_{month}"].astype(float)
    vv_db = df[f"VV_{month}"].astype(float)
    vh_lin = 10.0 ** (vh_db / 10.0)   # dB → linear power
    vv_lin = 10.0 ** (vv_db / 10.0)
    return (4.0 * vh_lin) / (vh_lin + vv_lin + EPS)


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
    # v6 additions — physics-motivated indices for aquaculture vs other water
    "SABI":         sabi,
    "CDOM":         cdom_proxy,
    "CI":           chlorophyll_index,
    "NDWI2":        ndwi2,
    "SAR_RVI":      sar_rvi,
}