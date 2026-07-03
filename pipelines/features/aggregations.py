"""
Temporal aggregations over 12 monthly values.
Takes a DataFrame of raw + index columns, returns a flat feature DataFrame.

Vectorized and optimized for high-performance extraction over large datasets.
"""

import numpy as np
import pandas as pd
from contracts.schema import MONTHS, ALL_BANDS, raw_col
from pipelines.features.indices import INDEX_FN_MAP


# ── Scalar aggregations ────────────────────────────────────────────────────────

def _agg_series(monthly_values: np.ndarray) -> dict[str, float]:
    """
    Given an array of monthly values for one band/index (potentially containing NaNs),
    return all scalar temporal aggregations over valid months.
    """
    valid_vals = monthly_values[~np.isnan(monthly_values)]
    if len(valid_vals) == 0:
        return {
            "mean":   0.0,
            "median": 0.0,
            "std":    0.0,
            "min":    0.0,
            "max":    0.0,
            "p10":    0.0,
            "p90":    0.0,
            "cv":     0.0,
            "range":  0.0,
        }

    mean = float(np.mean(valid_vals))
    std  = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0
    cv   = std / (abs(mean) + 1e-9)  # coefficient of variation; abs(mean) handles dB negatives

    return {
        "mean":   mean,
        "median": float(np.median(valid_vals)),
        "std":    std,
        "min":    float(np.min(valid_vals)),
        "max":    float(np.max(valid_vals)),
        "p10":    float(np.percentile(valid_vals, 10)),
        "p90":    float(np.percentile(valid_vals, 90)),
        "cv":     cv,
        "range":  float(np.max(valid_vals) - np.min(valid_vals)),
    }


# ── Persistence fractions (normalized counts) ──────────────────────────────────

PERSISTENCE_RULES: dict[str, tuple[str, str, float]] = {
    # feature_name          : (index_name, operator, threshold)
    "NDWI_pos_count":        ("NDWI",    ">",  0.0),
    "MNDWI_pos_count":       ("MNDWI",   ">",  0.0),
    "NDVI_low_count":        ("NDVI",    "<",  0.1),
    "AWEInsh_pos_count":     ("AWEInsh", ">",  0.0),
    "SAR_diff_neg15_count":  ("SAR_diff_db", "<", -15.0),  # very negative = strong water signal
}


def _persistence(monthly_values: np.ndarray, operator: str, threshold: float) -> float:
    """
    Computes the fraction of valid observed months that satisfy the threshold rule.
    """
    valid_vals = monthly_values[~np.isnan(monthly_values)]
    if len(valid_vals) == 0:
        return 0.0
    if operator == ">":
        return float(np.mean(valid_vals > threshold))
    elif operator == "<":
        return float(np.mean(valid_vals < threshold))
    elif operator == ">=":
        return float(np.mean(valid_vals >= threshold))
    elif operator == "<=":
        return float(np.mean(valid_vals <= threshold))
    raise ValueError(f"Unknown operator: {operator}")


# ── Consecutive-month change features ─────────────────────────────────────────

def _consecutive_changes(monthly_values: np.ndarray) -> dict[str, float]:
    """
    Computes statistics over consecutive-month absolute differences for valid months.
    """
    valid_vals = monthly_values[~np.isnan(monthly_values)]
    if len(valid_vals) <= 1:
        return {
            "max_consec_change":  0.0,
            "mean_consec_change": 0.0,
            "monotone_fraction":  1.0,
        }
    diffs = np.abs(np.diff(valid_vals))   # consecutive changes
    return {
        "max_consec_change":  float(np.max(diffs)),
        "mean_consec_change": float(np.mean(diffs)),
        "monotone_fraction":  float(np.mean(diffs < 0.05)),
    }


# Bands/indices to compute consecutive change features for
CONSEC_CHANGE_TARGETS = ["NDWI", "MNDWI", "VV", "NDTI", "re1_nir"]


# ── Main feature builder ───────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, seasonal_stats: dict | None = None) -> pd.DataFrame:
    """
    Transforms a raw dataframe (train or test) into a flat feature matrix.
    Vectorized implementation for maximum speed.
    """
    n = len(df)
    if n == 0:
        # Handle empty DataFrame gracefully
        cols = feature_names(exclude_id=False)
        return pd.DataFrame(columns=cols)

    # 1. Precompute indices for each month across all rows
    monthly_index_values: dict[str, pd.DataFrame] = {}
    for index_name, fn in INDEX_FN_MAP.items():
        cols_dict = {}
        for m in MONTHS:
            cols_dict[m] = fn(df, m)
        monthly_index_values[index_name] = pd.DataFrame(cols_dict, index=df.index)

    # Precompute raw bands for each month
    monthly_band_values: dict[str, pd.DataFrame] = {}
    for band in ALL_BANDS:
        cols = [raw_col(band, m) for m in MONTHS]
        monthly_band_values[band] = df[cols].rename(columns={raw_col(band, m): m for m in MONTHS}).astype(float)

    # 1b. Z-score normalize monthly values if seasonal_stats are provided
    monthly_index_values_norm = {}
    for index_name in INDEX_FN_MAP:
        df_norm = monthly_index_values[index_name].copy()
        if seasonal_stats and index_name in seasonal_stats:
            for m in MONTHS:
                mean = seasonal_stats[index_name][m]["mean"]
                std = seasonal_stats[index_name][m]["std"]
                df_norm[m] = (df_norm[m] - mean) / std
        monthly_index_values_norm[index_name] = df_norm

    monthly_band_values_norm = {}
    for band in ALL_BANDS:
        df_norm = monthly_band_values[band].copy()
        if seasonal_stats and band in seasonal_stats:
            for m in MONTHS:
                mean = seasonal_stats[band][m]["mean"]
                std = seasonal_stats[band][m]["std"]
                df_norm[m] = (df_norm[m] - mean) / std
        monthly_band_values_norm[band] = df_norm

    # 2. Compute window metadata features
    mask = ~monthly_band_values[ALL_BANDS[0]].isna()  # shape (n, 12)
    has_valid = mask.any(axis=1).values
    first_valid = np.argmax(mask.values, axis=1)
    valid_length = mask.sum(axis=1).values

    w_start = np.where(has_valid, first_valid + 1, 1)
    w_length = np.where(has_valid, valid_length, 12)
    w_center = w_start + (w_length - 1) / 2.0

    dict_features = {}
    dict_features["window_start"] = w_start.astype(float)
    dict_features["window_length"] = w_length.astype(float)
    dict_features["window_center"] = w_center
    dict_features["window_start_sin"] = np.sin(2 * np.pi * w_start / 12.0)
    dict_features["window_start_cos"] = np.cos(2 * np.pi * w_start / 12.0)
    dict_features["window_center_sin"] = np.sin(2 * np.pi * w_center / 12.0)
    dict_features["window_center_cos"] = np.cos(2 * np.pi * w_center / 12.0)

    # 3. Vectorized aggregations helper
    def add_aggregations(feats_dict: dict, source_df: pd.DataFrame, source_df_unnorm: pd.DataFrame, prefix: str):
        mean = source_df.mean(axis=1)
        std  = source_df.std(axis=1, ddof=1).fillna(0.0)
        
        # Compute CV on unnormalized values to prevent division by near-zero standardized means
        mean_unnorm = source_df_unnorm.mean(axis=1)
        std_unnorm  = source_df_unnorm.std(axis=1, ddof=1).fillna(0.0)
        cv          = std_unnorm / (mean_unnorm.abs() + 1e-9)
        
        min_val = source_df.min(axis=1)
        max_val = source_df.max(axis=1)

        feats_dict[f"{prefix}__mean"]   = mean.fillna(0.0)
        feats_dict[f"{prefix}__median"] = source_df.median(axis=1).fillna(0.0)
        feats_dict[f"{prefix}__std"]    = std
        feats_dict[f"{prefix}__min"]    = min_val.fillna(0.0)
        feats_dict[f"{prefix}__max"]    = max_val.fillna(0.0)
        feats_dict[f"{prefix}__p10"]    = source_df.quantile(0.1, axis=1).fillna(0.0)
        feats_dict[f"{prefix}__p90"]    = source_df.quantile(0.9, axis=1).fillna(0.0)
        feats_dict[f"{prefix}__cv"]     = cv.fillna(0.0)
        feats_dict[f"{prefix}__range"]  = (max_val - min_val).fillna(0.0)

    # Aggregations for raw bands
    for band in ALL_BANDS:
        add_aggregations(dict_features, monthly_band_values_norm[band], monthly_band_values[band], band)

    # Aggregations for spectral indices
    for index_name in INDEX_FN_MAP:
        add_aggregations(dict_features, monthly_index_values_norm[index_name], monthly_index_values[index_name], index_name)

    # 4. Vectorized Persistence fractions
    for feat_name, (index_name, operator, threshold) in PERSISTENCE_RULES.items():
        source_df = monthly_index_values[index_name]
        valid_counts = (~source_df.isna()).sum(axis=1)
        
        if operator == ">":
            meets_rule = (source_df > threshold).sum(axis=1)
        elif operator == "<":
            meets_rule = (source_df < threshold).sum(axis=1)
        else:
            raise ValueError(f"Unknown operator: {operator}")
            
        fraction = meets_rule / np.maximum(valid_counts, 1)
        dict_features[feat_name] = np.where(valid_counts > 0, fraction, 0.0)

    # 5. Vectorized Consecutive-month changes
    for target in CONSEC_CHANGE_TARGETS:
        if target in INDEX_FN_MAP:
            source_df = monthly_index_values_norm[target]
        else:
            source_df = monthly_band_values_norm[target]

        arr = source_df.values
        nan_mask = np.isnan(arr)
        
        diffs_dict = {}
        for i in range(n):
            valid_vals = arr[i, ~nan_mask[i]]
            stats_vals = _consecutive_changes(valid_vals)
            for k, v in stats_vals.items():
                if k not in diffs_dict:
                    diffs_dict[k] = []
                diffs_dict[k].append(v)
                
        for k, v in diffs_dict.items():
            dict_features[f"{target}__{k}"] = np.array(v, dtype=float)

    # 6. Vectorized Linear Trend Slopes
    t_coords = np.arange(1, 13, dtype=float)
    for target in CONSEC_CHANGE_TARGETS:
        if target in INDEX_FN_MAP:
            source_df = monthly_index_values_norm[target]
        else:
            source_df = monthly_band_values_norm[target]

        Y = source_df.values
        valid = ~np.isnan(Y)
        counts = valid.sum(axis=1)

        T_matrix = np.tile(t_coords, (n, 1))
        T_valid = np.where(valid, T_matrix, 0.0)
        Y_valid = np.where(valid, Y, 0.0)

        mean_t = T_valid.sum(axis=1) / np.maximum(counts, 1)
        mean_y = Y_valid.sum(axis=1) / np.maximum(counts, 1)

        dt = np.where(valid, T_matrix - mean_t[:, None], 0.0)
        dy = np.where(valid, Y - mean_y[:, None], 0.0)

        num = (dt * dy).sum(axis=1)
        den = (dt * dt).sum(axis=1)

        slope = np.where((counts > 1) & (den > 1e-9), num / den, 0.0)
        dict_features[f"{target}__trend_slope"] = np.nan_to_num(slope, 0.0)

    # 7. Vectorized Cross-index agreement
    ndwi_df  = monthly_index_values["NDWI"]
    mndwi_df = monthly_index_values["MNDWI"]
    awei_df  = monthly_index_values["AWEInsh"]

    all_positive = (ndwi_df > 0) & (mndwi_df > 0) & (awei_df > 0)
    all_negative = (ndwi_df <= 0) & (mndwi_df <= 0) & (awei_df <= 0)

    valid_counts = (~ndwi_df.isna()).sum(axis=1)

    row_agreement = all_positive.sum(axis=1) / valid_counts.replace(0, 1)
    row_unanimous = (all_positive | all_negative).sum(axis=1) / valid_counts.replace(0, 1)

    dict_features["water_index_agreement"] = row_agreement.fillna(0.0)
    dict_features["water_index_unanimous"] = row_unanimous.fillna(0.0)

    # 8. ID passthrough and assembly
    result_features = pd.DataFrame(dict_features, index=df.index)
    result = pd.concat([df[["ID"]].reset_index(drop=True),
                        result_features.reset_index(drop=True)], axis=1)
    return result


def feature_names(exclude_id: bool = True) -> list[str]:
    """
    Returns the ordered list of feature column names this pipeline produces.
    Use this to validate the feature matrix contract downstream.
    """
    cols = []
    
    # Metadata features first
    cols.extend([
        "window_start",
        "window_length",
        "window_center",
        "window_start_sin",
        "window_start_cos",
        "window_center_sin",
        "window_center_cos",
    ])

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

    for target in CONSEC_CHANGE_TARGETS:
        cols.append(f"{target}__trend_slope")

    cols.append("water_index_agreement")
    cols.append("water_index_unanimous")

    if not exclude_id:
        cols = ["ID"] + cols

    return cols


def fit_seasonal_stats(df: pd.DataFrame) -> dict:
    """
    Computes monthly mean and standard deviation for each raw band and spectral index.
    Ignores NaNs (e.g. masked months in the test set).
    """
    stats = {}
    
    # Precompute indices for each month across all rows
    monthly_indices = {}
    for idx_name, fn in INDEX_FN_MAP.items():
        cols = {}
        for m in MONTHS:
            cols[m] = fn(df, m)
        monthly_indices[idx_name] = pd.DataFrame(cols, index=df.index)

    # Stats for raw bands
    for band in ALL_BANDS:
        stats[band] = {}
        for m in MONTHS:
            col = raw_col(band, m)
            vals = df[col].dropna()
            stats[band][m] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
            }

    # Stats for spectral indices
    for idx_name in INDEX_FN_MAP:
        stats[idx_name] = {}
        for m in MONTHS:
            vals = monthly_indices[idx_name][m].dropna()
            stats[idx_name][m] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
            }

    return stats

