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

# -- Seasonal shape tests (v6.2) -----------------------------------------------

from pipelines.features.aggregations import _seasonal_shape, SEASONAL_SHAPE_TARGETS


def test_seasonal_shape_output_keys():
    vals = np.arange(12, dtype=float)
    result = _seasonal_shape(vals)
    assert set(result.keys()) == {'peak_month', 'trough_month', 'seasonal_amplitude'}


def test_seasonal_shape_peak_trough():
    vals = np.zeros(12)
    vals[5] = 1.0
    vals[11] = -1.0
    result = _seasonal_shape(vals)
    assert result['peak_month']   == pytest.approx(5.0)
    assert result['trough_month'] == pytest.approx(11.0)


def test_seasonal_shape_amplitude():
    vals = np.arange(12, dtype=float)
    result = _seasonal_shape(vals)
    assert result['seasonal_amplitude'] == pytest.approx(9.0)


def test_seasonal_shape_stable_near_zero():
    result = _seasonal_shape(np.full(12, 0.5))
    assert abs(result['seasonal_amplitude']) < 1e-9


def test_seasonal_shape_discriminates_wetland_vs_pond():
    wetland = np.array([-0.3,-0.2,-0.1,0.0,0.1,0.5,0.6,0.5,0.2,-0.1,-0.2,-0.3])
    pond    = np.full(12, 0.4)
    assert _seasonal_shape(wetland)['seasonal_amplitude'] > _seasonal_shape(pond)['seasonal_amplitude']
    assert _seasonal_shape(wetland)['seasonal_amplitude'] > 0.5


def test_seasonal_shape_in_feature_matrix(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for target in SEASONAL_SHAPE_TARGETS:
        for suffix in ['peak_month', 'trough_month', 'seasonal_amplitude']:
            col = f'{target}__{suffix}'
            assert col in result.columns, f'Missing: {col}'
            assert not result[col].isna().any(), f'NaN in: {col}'


def test_seasonal_shape_in_feature_names():
    names = feature_names()
    for target in SEASONAL_SHAPE_TARGETS:
        for suffix in ['peak_month', 'trough_month', 'seasonal_amplitude']:
            assert f'{target}__{suffix}' in names


def test_seasonal_shape_month_indices_in_range(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for target in SEASONAL_SHAPE_TARGETS:
        assert result[f'{target}__peak_month'].between(0, 11).all()
        assert result[f'{target}__trough_month'].between(0, 11).all()


# -- Fourier harmonic tests (v6.3) ---------------------------------------------

from pipelines.features.aggregations import _fourier_harmonics, FOURIER_TARGETS


def test_fourier_harmonics_output_keys():
    vals = np.arange(12, dtype=float)
    result = _fourier_harmonics(vals)
    assert set(result.keys()) == {'harmonic_A1', 'harmonic_phi1', 'harmonic_A2', 'harmonic_phi2'}


def test_fourier_harmonics_flat_signal_near_zero():
    '''Flat signal -> no annual or semi-annual cycle -> both amplitudes near 0.'''
    result = _fourier_harmonics(np.full(12, 0.5))
    assert abs(result['harmonic_A1']) < 1e-9
    assert abs(result['harmonic_A2']) < 1e-9


def test_fourier_harmonics_pure_annual_sine():
    '''Pure annual sine -> A1 > 0, A2 near 0.'''
    t = np.arange(12)
    vals = np.sin(2 * np.pi * t / 12)  # one full cycle
    result = _fourier_harmonics(vals)
    assert result['harmonic_A1'] == pytest.approx(1.0, abs=1e-6)
    assert abs(result['harmonic_A2']) < 1e-6


def test_fourier_harmonics_pure_semi_annual_sine():
    '''Pure semi-annual sine -> A2 > 0, A1 near 0.'''
    t = np.arange(12)
    vals = np.sin(2 * np.pi * 2 * t / 12)  # two full cycles
    result = _fourier_harmonics(vals)
    assert abs(result['harmonic_A1']) < 1e-6
    assert result['harmonic_A2'] == pytest.approx(1.0, abs=1e-6)


def test_fourier_harmonics_pond_vs_wetland():
    '''Permanent pond (flat) has lower A1 than seasonal wetland.'''
    wetland = np.array([-0.3,-0.2,-0.1,0.0,0.1,0.5,0.6,0.5,0.2,-0.1,-0.2,-0.3])
    pond    = np.full(12, 0.4)
    assert _fourier_harmonics(wetland)['harmonic_A1'] > _fourier_harmonics(pond)['harmonic_A1']


def test_fourier_harmonics_phi_range():
    '''Phase values must be in [-pi, pi].'''
    vals = np.random.default_rng(42).uniform(-1, 1, 12)
    result = _fourier_harmonics(vals)
    assert -np.pi <= result['harmonic_phi1'] <= np.pi
    assert -np.pi <= result['harmonic_phi2'] <= np.pi


def test_fourier_harmonics_in_feature_matrix(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for target in FOURIER_TARGETS:
        for suffix in ['harmonic_A1', 'harmonic_phi1', 'harmonic_A2', 'harmonic_phi2']:
            col = f'{target}__{suffix}'
            assert col in result.columns, f'Missing: {col}'
            assert not result[col].isna().any(), f'NaN in: {col}'


def test_fourier_harmonics_in_feature_names():
    names = feature_names()
    for target in FOURIER_TARGETS:
        for suffix in ['harmonic_A1', 'harmonic_phi1', 'harmonic_A2', 'harmonic_phi2']:
            assert f'{target}__{suffix}' in names


def test_fourier_harmonics_amplitude_nonnegative(minimal_raw_df, minimal_regions):
    result = build_feature_matrix(minimal_raw_df, minimal_regions)
    for target in FOURIER_TARGETS:
        assert (result[f'{target}__harmonic_A1'] >= 0).all()
        assert (result[f'{target}__harmonic_A2'] >= 0).all()
