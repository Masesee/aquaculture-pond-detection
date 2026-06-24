"""
Temporal aggregations over 12 monthly values.
Takes a DataFrame of raw + index columns, returns a flat feature DataFrame.

Every aggregation is deterministic: same input → same output, always.
The feature names produced here are the contract downstream models consume.
"""

import numpy as np
import pandas as pd
from contracts.schema import MONTHS, ALL_BANDS, raw_col
from pipelines.features.indices import INDEX_FN_MAP


# ── Scalar aggregations ────────────────────────────────────────────────────────

def _agg_series(monthly_values: np.ndarray) -> dict[str, float]:
    """
    Given a (12,) array of monthly values for one band/index,
    return all scalar temporal aggregations.
    """
    mean   = float(np.mean(monthly_values))
    std    = float(np.std(monthly_values, ddof=1)) if len(monthly_values) > 1 else 0.0
    cv     = std / (abs(mean) + 1e-9)  # coefficient of variation; abs(mean) handles dB negatives

    return {
        "mean":   mean,
        "median": float(np.median(monthly_values)),
        "std":    std,
        "min":    float(np.min(monthly_values)),
        "max":    float(np.max(monthly_values)),
        "p10":    float(np.percentile(monthly_values, 10)),
        "p90":    float(np.percentile(monthly_values, 90)),
        "cv":     cv,
        "range":  float(np.max(monthly_values) - np.min(monthly_values)),
    }


# ── Persistence counts ─────────────────────────────────────────────────────────

PERSISTENCE_RULES: dict[str, tuple[str, str, float]] = {
    # feature_name          : (index_name, operator, threshold)
    "NDWI_pos_count":        ("NDWI",    ">",  0.0),
    "MNDWI_pos_count":       ("MNDWI",   ">",  0.0),
    "NDVI_low_count":        ("NDVI",    "<",  0.1),
    "AWEInsh_pos_count":     ("AWEInsh", ">",  0.0),
    "SAR_diff_neg15_count":  ("SAR_diff_db", "<", -15.0),  # very negative = strong water signal
    "SABI_pos_count":        ("SABI",    ">",  0.0),        # months where nir > red = productive water
    "SAR_RVI_low_count":     ("SAR_RVI", "<",  0.25),      # months with very low RVI = open water SAR signal
}


def _persistence(monthly_values: np.ndarray, operator: str, threshold: float) -> int:
    if operator == ">":
        return int(np.sum(monthly_values > threshold))
    elif operator == "<":
        return int(np.sum(monthly_values < threshold))
    elif operator == ">=":
        return int(np.sum(monthly_values >= threshold))
    elif operator == "<=":
        return int(np.sum(monthly_values <= threshold))
    raise ValueError(f"Unknown operator: {operator}")


# ── Consecutive-month change features ─────────────────────────────────────────

def _consecutive_changes(monthly_values: np.ndarray) -> dict[str, float]:
    """
    Computes statistics over 11 consecutive-month absolute differences.
    Measures temporal stability — invariant across time periods.
    """
    diffs = np.abs(np.diff(monthly_values))   # shape (11,)
    return {
        "max_consec_change":  float(np.max(diffs)),
        "mean_consec_change": float(np.mean(diffs)),
        "monotone_fraction":  float(np.mean(diffs < 0.05)),
    }


# Bands/indices to compute consecutive change features for
# v5 originals: NDWI, MNDWI, VV, NDTI, re1_nir
# v6 additions: SABI (algal bloom temporal stability), CI (chlorophyll-a cycles)
# v6.2 additions: NDWI2, SAR_RVI (top-SHAP v6.1 indices, temporally stable diffs)
CONSEC_CHANGE_TARGETS = ["NDWI", "MNDWI", "VV", "NDTI", "re1_nir", "SABI", "CI",
                         "NDWI2", "SAR_RVI"]


# ── Seasonal shape features ──────────────────────────────────────────────────

def _seasonal_shape(monthly_values: np.ndarray) -> dict[str, float]:
    """
    Temporally invariant seasonal shape features.

    Captures WHEN and HOW MUCH an index varies over the year without
    encoding absolute reflectance or calendar-specific states. Safe
    to use across training and test periods that cover different years.

    Returns
    -------
    peak_month        : argmax index (0–11) — which month has the highest value.
                        Physical meaning: timing of peak water / productivity.
                        Transfers across years: a pond peaks randomly (always wet),
                        a seasonal wetland peaks consistently in summer.
    trough_month      : argmin index (0–11) — month with the lowest value.
    seasonal_amplitude: mean(top-3 months) − mean(bottom-3 months).
                        More robust than range (max−min) — ignores single-month
                        cloud artefacts. Large = highly seasonal, small = stable.
    """
    sorted_vals  = np.sort(monthly_values)
    top3_mean    = float(np.mean(sorted_vals[-3:]))
    bottom3_mean = float(np.mean(sorted_vals[:3]))
    return {
        "peak_month":         float(np.argmax(monthly_values)),
        "trough_month":       float(np.argmin(monthly_values)),
        "seasonal_amplitude": top3_mean - bottom3_mean,
    }


# Indices to compute seasonal shape features for.
# Selected: highest SHAP contributors from v6.1 that are most affected by
# seasonal variability confounding pond vs wetland discrimination.
# Quarterly bins (Sub 27) on these same indices caused temporal overfitting;
# seasonal_amplitude and peak_month are invariant because they measure SHAPE
# not absolute value in a fixed calendar window.
SEASONAL_SHAPE_TARGETS = ["NDWI", "MNDWI", "NDTI", "NDWI2", "SAR_RVI", "re1_nir", "SABI"]


# ── Fourier harmonic features (v6.3) ──────────────────────────────────────

def _fourier_harmonics(monthly_values: np.ndarray) -> dict[str, float]:
    """
    Extract Fourier harmonic features from 12 monthly observations.

    Uses numpy real FFT on N=12 evenly-spaced monthly samples.
    Returns amplitude and phase for the first two harmonics:

    A1 / phi1 : annual cycle (period = 12 months)
        A1  = strength of the dominant yearly seasonal swing.
              Near zero for permanent ponds (flat signal).
              Large for seasonal wetlands (wet/dry cycle).
        phi1 = phase in radians [-pi, pi] — when the annual peak occurs.
              Invariant across years: a summer-wet wetland always has phi1 ~ pi/2.

    A2 / phi2 : semi-annual cycle (period = 6 months)
        A2  = strength of sub-annual periodicity.
              Fires for aquaculture ponds with two harvest/drain cycles per year.

    Temporal invariance: A1 and A2 measure cycle STRENGTH (not timing).
    They are safe across training/test periods from different years.

    IMPORTANT — phases (phi1, phi2) are NOT included.
    Lesson 19: phase encodes WHEN the annual/semi-annual peak occurs within
    the calendar year. Interannual variability in monsoon onset, drought
    timing etc. shifts phi by weeks year-to-year, causing the same
    failure as quarterly bins (Sub 27, Sub 32). Amplitudes only.

    Normalization: amplitudes scaled by 2/N to match original signal units.
    """
    fft = np.fft.rfft(monthly_values)  # length 7 for N=12 (indices 0..6)
    scale = 2.0 / len(monthly_values)  # = 2/12
    return {
        "harmonic_A1": float(np.abs(fft[1]) * scale),
        "harmonic_A2": float(np.abs(fft[2]) * scale),
    }


# Indices to compute Fourier harmonic features for.
# Selected: the 5 highest SHAP contributors in v6.1 that carry seasonal signal.
# NDWI/MNDWI: primary water detection. NDTI: turbidity seasonality.
# SAR_RVI: radar vegetation cycle. re1_nir: canopy phenology.
# Applying to SABI/NDWI2 would add partial redundancy without new physics.
FOURIER_TARGETS = ["NDWI", "MNDWI", "NDTI", "SAR_RVI", "re1_nir"]



# -- Spatial features ----------------------------------------------------------


# Pond cluster centroid — derived from region_map visual inspection.
# The dense orange cluster sits around lon=48.85, lat=39.48.
# Used to compute a distance proxy that is time-invariant.
# POND_CLUSTER_CENTROID = (48.85, 39.48)


# def _distance_to_centroid(lon: pd.Series, lat: pd.Series) -> pd.Series:
#     """
#     Log-transformed distance to pond cluster centroid.
#     log1p compresses large distances, reducing leverage of OOD points.
#     """
#     dlon = lon - POND_CLUSTER_CENTROID[0]
#     dlat = lat - POND_CLUSTER_CENTROID[1]
#     raw_dist = np.sqrt(dlon**2 + dlat**2)
#     return np.log1p(raw_dist)


# ── Main feature builder ───────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, region_series: pd.Series) -> pd.DataFrame:
    """
    Transforms a raw dataframe (train or test) into a flat feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with all band columns + ID, lon, lat.
    region_series : pd.Series
        Integer region label (0 or 1) aligned with df.index.
        Produced by pipelines.eda.regional_analysis.assign_regions.

    Returns
    -------
    pd.DataFrame
        One row per location. Columns: ID + all engineered features.
        No label column — caller appends it if needed.
    """
    n = len(df)
    feature_rows = []

    # Pre-compute all monthly index arrays: shape (n_locations, 12)
    monthly_index_values: dict[str, np.ndarray] = {}
    for index_name, fn in INDEX_FN_MAP.items():
        monthly_cols = np.column_stack([
            fn(df, m).values for m in MONTHS
        ])  # shape (n, 12)
        monthly_index_values[index_name] = monthly_cols

    # Pre-compute all monthly raw band arrays
    monthly_band_values: dict[str, np.ndarray] = {}
    for band in ALL_BANDS:
        monthly_cols = np.column_stack([
            df[raw_col(band, m)].astype(float).values for m in MONTHS
        ])  # shape (n, 12)
        monthly_band_values[band] = monthly_cols

    for i in range(n):
        row: dict[str, float | int] = {}

        # ── Raw band aggregations ──
        for band in ALL_BANDS:
            vals = monthly_band_values[band][i]
            aggs = _agg_series(vals)
            for agg_name, agg_val in aggs.items():
                row[f"{band}__{agg_name}"] = agg_val

        # ── Spectral index aggregations ──
        for index_name in INDEX_FN_MAP:
            vals = monthly_index_values[index_name][i]
            aggs = _agg_series(vals)
            for agg_name, agg_val in aggs.items():
                row[f"{index_name}__{agg_name}"] = agg_val

        # ── Persistence counts ──
        for feat_name, (index_name, operator, threshold) in PERSISTENCE_RULES.items():
            vals = monthly_index_values[index_name][i]
            row[feat_name] = _persistence(vals, operator, threshold)

        # ── Consecutive-month change features ──
        for target in CONSEC_CHANGE_TARGETS:
            if target in INDEX_FN_MAP:
                vals = monthly_index_values[target][i]
            else:
                vals = monthly_band_values[target][i]
            changes = _consecutive_changes(vals)
            for change_name, change_val in changes.items():
                row[f"{target}__{change_name}"] = change_val

        # ── Seasonal shape features (v6.2) ──
        for target in SEASONAL_SHAPE_TARGETS:
            vals = monthly_index_values[target][i]
            shape = _seasonal_shape(vals)
            for shape_name, shape_val in shape.items():
                row[f"{target}__{shape_name}"] = shape_val

        # ── Fourier harmonic features (v6.3) ──
        for target in FOURIER_TARGETS:
            vals = monthly_index_values[target][i]
            harmonics = _fourier_harmonics(vals)
            for h_name, h_val in harmonics.items():
                row[f"{target}__{h_name}"] = h_val

        # ── Cross-index water agreement ──
        ndwi_monthly  = monthly_index_values["NDWI"][i]
        mndwi_monthly = monthly_index_values["MNDWI"][i]
        awei_monthly  = monthly_index_values["AWEInsh"][i]

        all_positive = (ndwi_monthly > 0) & (mndwi_monthly > 0) & (awei_monthly > 0)
        all_negative = (ndwi_monthly <= 0) & (mndwi_monthly <= 0) & (awei_monthly <= 0)

        row["water_index_agreement"]  = float(np.mean(all_positive))
        row["water_index_unanimous"]  = float(np.mean(all_positive | all_negative))

        feature_rows.append(row)

    features = pd.DataFrame(feature_rows, index=df.index)

    # ── Spatial features ──
    # features["dist_to_pond_centroid"] = _distance_to_centroid(
    #     df["lon"], df["lat"]
    # ).values
    features["region"] = region_series.values

    # ── ID passthrough ──
    result = pd.concat([df[["ID"]].reset_index(drop=True),
                        features.reset_index(drop=True)], axis=1)
    return result


def feature_names(exclude_id: bool = True) -> list[str]:
    """
    Returns the ordered list of feature column names this pipeline produces.
    Use this to validate the feature matrix contract downstream.
    """
    cols = []
    agg_suffixes = ["mean", "median", "std", "min", "max", "p10", "p90", "cv", "range"]

    for band in ALL_BANDS:
        for suf in agg_suffixes:
            cols.append(f"{band}__{suf}")

    for index_name in INDEX_FN_MAP:
        for suf in agg_suffixes:
            cols.append(f"{index_name}__{suf}")

    for feat_name in PERSISTENCE_RULES:
        cols.append(feat_name)

    for target in CONSEC_CHANGE_TARGETS:
        for suffix in ["max_consec_change", "mean_consec_change", "monotone_fraction"]:
            cols.append(f"{target}__{suffix}")

    for target in SEASONAL_SHAPE_TARGETS:
        for suffix in ["peak_month", "trough_month", "seasonal_amplitude"]:
            cols.append(f"{target}__{suffix}")

    for target in FOURIER_TARGETS:
        for suffix in ["harmonic_A1", "harmonic_A2"]:
            cols.append(f"{target}__{suffix}")

    cols.append("water_index_agreement")
    cols.append("water_index_unanimous")

    # cols.append("dist_to_pond_centroid")
    cols.append("region")

    if not exclude_id:
        cols = ["ID"] + cols

    return cols
