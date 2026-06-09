"""
Gate tests for CV strategy.
Deterministic. No real data. Must pass in < 2s.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from contracts.schema import TARGET_COL
from pipelines.training.cv_strategy import make_strata, make_cv_splits, describe_splits


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """
    80-row dataframe mimicking the real class/region distribution:
    region 0: 27 rows, 4% pond  (~1 pond)
    region 1: 53 rows, 59% pond (~31 ponds)
    """
    rng = np.random.default_rng(42)
    n0, n1 = 27, 53

    labels_r0 = np.array([1] + [0] * (n0 - 1))   # 1 pond in region 0
    labels_r1 = np.array([1] * 31 + [0] * (n1 - 31))

    labels  = np.concatenate([labels_r0, labels_r1])
    regions = np.array([0] * n0 + [1] * n1)

    df = pd.DataFrame({
        TARGET_COL: labels,
        "region":   regions,
        "feature1": rng.standard_normal(n0 + n1),
    })
    return df


def test_make_strata_values(synthetic_train_df):
    strata = make_strata(synthetic_train_df[TARGET_COL], synthetic_train_df["region"])
    assert set(strata.unique()).issubset({0, 1, 2, 3})


def test_make_strata_encoding(synthetic_train_df):
    """Stratum 3 = pond (label=1) AND region 1."""
    strata = make_strata(synthetic_train_df[TARGET_COL], synthetic_train_df["region"])
    is_stratum_3 = strata == 3
    is_pond_r1   = (synthetic_train_df[TARGET_COL] == 1) & (synthetic_train_df["region"] == 1)
    assert (is_stratum_3 == is_pond_r1).all()


def test_cv_splits_count(synthetic_train_df):
    splits = make_cv_splits(synthetic_train_df, n_splits=5)
    assert len(splits) == 5


def test_cv_splits_cover_all_rows(synthetic_train_df):
    """Every row appears in exactly one validation fold."""
    n = len(synthetic_train_df)
    splits = make_cv_splits(synthetic_train_df, n_splits=5)
    all_val_indices = np.concatenate([val for _, val in splits])
    assert len(all_val_indices) == n
    assert len(set(all_val_indices)) == n


def test_cv_splits_train_val_disjoint(synthetic_train_df):
    splits = make_cv_splits(synthetic_train_df, n_splits=5)
    for fold, (tr, val) in enumerate(splits):
        overlap = set(tr) & set(val)
        assert len(overlap) == 0, f"Fold {fold}: train/val overlap on indices {overlap}"


def test_describe_splits_shape(synthetic_train_df):
    splits = make_cv_splits(synthetic_train_df, n_splits=5)
    summary = describe_splits(synthetic_train_df, splits)
    assert summary.shape == (5, 7)


def test_describe_splits_val_sizes_sum(synthetic_train_df):
    splits = make_cv_splits(synthetic_train_df, n_splits=5)
    summary = describe_splits(synthetic_train_df, splits)
    assert summary["n_val"].sum() == len(synthetic_train_df)


def test_cv_reproducible(synthetic_train_df):
    """Same random_state → identical splits."""
    splits_a = make_cv_splits(synthetic_train_df, n_splits=5, random_state=42)
    splits_b = make_cv_splits(synthetic_train_df, n_splits=5, random_state=42)
    for (tr_a, val_a), (tr_b, val_b) in zip(splits_a, splits_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(val_a, val_b)