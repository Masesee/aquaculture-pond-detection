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
CONSEC_CHANGE_TARGETS = ["NDWI", "MNDWI", "VV", "NDTI", "re1_nir"]


# ── Spatial features ───────────────────────────────────────────────────────────

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

    cols.append("water_index_agreement")
    cols.append("water_index_unanimous")

    # cols.append("dist_to_pond_centroid")
    cols.append("region")

    if not exclude_id:
        cols = ["ID"] + cols

    return cols
