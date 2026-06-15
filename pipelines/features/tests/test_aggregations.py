"""
Gate tests for temporal aggregation pipeline.
Deterministic. No real data. Must pass in < 2s.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from contracts.schema import ALL_BANDS, MONTHS, raw_col
from pipelines.features.aggregations import (
    build_feature_matrix,
    feature_names,
    _agg_series,
    _persistence,
    PERSISTENCE_RULES,
    _consecutive_changes,
    CONSEC_CHANGE_TARGETS,
    _quarter_aggs,
    QUARTER_TARGETS,
    QUARTER_SLICES,
)


@pytest.fixture
def minimal_raw_df() -> pd.DataFrame:
    """Minimal synthetic dataframe: 6 rows, all required columns."""
    rng = np.random.default_rng(7)
    n = 6
    data = {
        "ID":  [f"ID_TR_{i:04d}" for i in range(n)],
        "lon": rng.uniform(48.0, 49.5, n),
        "lat": rng.uniform(39.0, 40.5, n),
    }
    for band in ALL_BANDS:
        for month in MONTHS:
            if band in ["VH", "VV"]:
                data[raw_col(band, month)] = rng.uniform(-28, -8, n)
            else:
                data[raw_col(band, month)] = rng.uniform(300, 6000, n)
    return pd.DataFrame(data)


@pytest.fixture
def minimal_regions(minimal_raw_df) -> pd.Series:
    return pd.Series([0, 0, 0, 1, 1, 1], index=minimal_raw_df.index)


# ── _agg_series unit tests ─────────────────────────────────────────────────────

def test_agg_series_keys():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    result = _agg_series(vals)
    expected_keys = {"mean", "median", "std", "min", "max", "p10", "p90", "cv", "range"}
    assert set(result.keys()) == expected_keys


def test_agg_series_known_values():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    result = _agg_series(vals)
    assert np.isclose(result["mean"],   6.5)
    assert np.isclose(result["min"],    1.0)
    assert np.isclose(result["max"],    12.0)
    assert np.isclose(result["range"],  11.0)
    assert result["cv"] > 0


def test_agg_series_constant():
    """Constant time series should have std=0, cv≈0."""
    vals = np.full(12, 5.0)
    result = _agg_series(vals)
    assert np.isclose(result["std"], 0.0, atol=1e-9)
    assert result["cv"] < 1e-6


# ── _persistence unit tests ────────────────────────────────────────────────────

def test_persistence_all_positive():
    vals = np.array([0.1, 0.2, 0.3] * 4)
    assert _persistence(vals, ">", 0.0) == 12


def test_persistence_none():
    vals = np.array([-0.1, -0.2, -0.3] * 4)
    assert _persistence(vals, ">", 0.0) == 0


def test_persistence_partial():
    vals = np.array([0.1, -0.1] * 6)
    assert _persistence(vals, ">", 0.0) == 6


def test_persistence_less_than():
    vals = np.array([-20.0] * 8 + [-10.0] * 4)
    assert _persistence(vals, "<", -15.0) == 8


# ── build_feature_matrix integration tests ────────────────────────────────────

def test_feature_matrix_shape(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    expected_feat_count = len(feature_names()) + 1  # +1 for ID column
    assert result.shape == (6, expected_feat_count), (
        f"Expected shape (6, {expected_feat_count}), got {result.shape}"
    )


def test_feature_matrix_no_nan(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    feat_cols = [c for c in result.columns if c != "ID"]
    assert not result[feat_cols].isna().any().any(), "Feature matrix contains NaN values"


def test_feature_matrix_no_inf(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    feat_cols = [c for c in result.columns if c != "ID"]
    assert not np.isinf(result[feat_cols].values).any(), "Feature matrix contains Inf values"


def test_feature_matrix_id_preserved(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    assert list(result["ID"]) == list(minimal_raw_df["ID"])


def test_feature_names_contract(minimal_raw_df, minimal_regions):
    """Column order from build_feature_matrix must match feature_names() contract."""
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    actual_feat_cols   = [c for c in result.columns if c != "ID"]
    expected_feat_cols = feature_names(exclude_id=True)
    assert actual_feat_cols == expected_feat_cols, (
        f"Column order mismatch.\n"
        f"First diff at index: "
        f"{next(i for i,(a,e) in enumerate(zip(actual_feat_cols,expected_feat_cols)) if a!=e)}"
    )


def test_region_values_binary(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    assert set(result["region"].unique()).issubset({0, 1})


def test_persistence_counts_integer(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for feat in PERSISTENCE_RULES:
        col = result[feat]
        assert col.dtype in [np.int64, np.int32, int, np.int_], (
            f"Persistence feature {feat} should be integer, got {col.dtype}"
        )
        assert col.between(0, 12).all(), (
            f"Persistence feature {feat} out of [0,12]: {col.values}"
        )


def test_consecutive_changes_constant():
    """Constant series: all diffs=0, monotone_fraction=1."""
    vals = np.full(12, 0.15)
    result = _consecutive_changes(vals)
    assert np.isclose(result["max_consec_change"],  0.0,  atol=1e-9)
    assert np.isclose(result["mean_consec_change"], 0.0,  atol=1e-9)
    assert np.isclose(result["monotone_fraction"],  1.0,  atol=1e-9)


def test_consecutive_changes_volatile():
    """Alternating 0/1 series: all diffs=1, monotone_fraction=0."""
    vals = np.array([0.0, 1.0] * 6)
    result = _consecutive_changes(vals)
    assert np.isclose(result["max_consec_change"],  1.0, atol=1e-9)
    assert np.isclose(result["mean_consec_change"], 1.0, atol=1e-9)
    assert np.isclose(result["monotone_fraction"],  0.0, atol=1e-9)


def test_consecutive_changes_output_keys():
    vals = np.random.default_rng(0).standard_normal(12)
    result = _consecutive_changes(vals)
    assert set(result.keys()) == {"max_consec_change", "mean_consec_change", "monotone_fraction"}


def test_water_index_agreement_in_feature_matrix(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    assert "water_index_agreement" in result.columns
    assert "water_index_unanimous" in result.columns
    assert result["water_index_agreement"].between(0.0, 1.0).all()
    assert result["water_index_unanimous"].between(0.0, 1.0).all()


def test_consec_change_features_in_matrix(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for target in CONSEC_CHANGE_TARGETS:
        for suffix in ["max_consec_change", "mean_consec_change", "monotone_fraction"]:
            col = f"{target}__{suffix}"
            assert col in result.columns, f"Missing: {col}"
            assert not result[col].isna().any()


# ── Quarter aggregation tests ────────────────────────────────────────────────────────

def test_quarter_aggs_output_keys():
    """_quarter_aggs must return exactly 8 keys: Q1-Q4 x mean/max."""
    vals = np.arange(12, dtype=float)  # 0..11
    result = _quarter_aggs(vals)
    expected_keys = {
        "Q1_mean", "Q1_max",
        "Q2_mean", "Q2_max",
        "Q3_mean", "Q3_max",
        "Q4_mean", "Q4_max",
    }
    assert set(result.keys()) == expected_keys


def test_quarter_aggs_correct_slicing():
    """
    vals = [0,1,2, 3,4,5, 6,7,8, 9,10,11]
    Q1 (months 0-2) mean = 1.0, max = 2.0
    Q3 (months 6-8) mean = 7.0, max = 8.0
    """
    vals = np.arange(12, dtype=float)
    result = _quarter_aggs(vals)
    assert result["Q1_mean"] == pytest.approx(1.0)
    assert result["Q1_max"]  == pytest.approx(2.0)
    assert result["Q3_mean"] == pytest.approx(7.0)
    assert result["Q3_max"]  == pytest.approx(8.0)
    assert result["Q4_mean"] == pytest.approx(10.0)
    assert result["Q4_max"]  == pytest.approx(11.0)


def test_quarter_aggs_seasonal_contrast():
    """
    Year-round water: all 12 months positive NDWI.
    All quarter means should be positive.
    Seasonal water: only Q3 positive (wet season), Q1 negative (dry).
    Q1_mean < 0, Q3_mean > 0 — the model can now distinguish them.
    """
    # Year-round pond
    pond_vals = np.full(12, 0.4)
    pond_q = _quarter_aggs(pond_vals)
    assert pond_q["Q1_mean"] > 0
    assert pond_q["Q3_mean"] > 0

    # Seasonal wetland: positive only in months 6-8 (Q3), negative otherwise
    wetland_vals = np.array([-0.2, -0.2, -0.2,
                             -0.1, -0.1, 0.1,
                              0.5,  0.5,  0.5,
                             0.1, -0.1, -0.2])
    wetland_q = _quarter_aggs(wetland_vals)
    assert wetland_q["Q1_mean"] < 0, "Dry-season wetland should have negative Q1 NDWI"
    assert wetland_q["Q3_mean"] > 0, "Wet-season wetland should have positive Q3 NDWI"
    # The contrast (Q3 - Q1) should be large
    contrast = wetland_q["Q3_mean"] - wetland_q["Q1_mean"]
    assert contrast > 0.5, f"Expected large seasonal contrast, got {contrast:.3f}"


def test_quarter_features_in_matrix(minimal_raw_df, minimal_regions):
    """All quarter columns for all QUARTER_TARGETS must be present and NaN-free."""
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for target in QUARTER_TARGETS:
        for q_name in QUARTER_SLICES:
            for stat in ["mean", "max"]:
                col = f"{target}__{q_name}_{stat}"
                assert col in result.columns, f"Missing quarter feature: {col}"
                assert not result[col].isna().any(), f"NaN in: {col}"


def test_quarter_features_in_feature_names():
    """feature_names() must include all quarter features."""
    names = feature_names()
    for target in QUARTER_TARGETS:
        for q_name in QUARTER_SLICES:
            for stat in ["mean", "max"]:
                expected = f"{target}__{q_name}_{stat}"
                assert expected in names, f"Missing from feature_names(): {expected}"