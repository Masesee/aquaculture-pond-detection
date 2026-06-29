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
from pipelines.training.cv_strategy import make_cv_splits, describe_splits


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """
    80-row dataframe mimicking the real class distribution.
    Includes ID and TARGET_COL.
    """
    rng = np.random.default_rng(42)
    n = 80
    labels = rng.choice([0, 1], size=n)
    df = pd.DataFrame({
        "ID": [f"ID_TR_NEW_{i:04d}" for i in range(n)],
        TARGET_COL: labels,
        "feature1": rng.standard_normal(n),
    })
    return df


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
    assert summary.shape == (5, 4)


def test_describe_splits_val_sizes_sum(synthetic_train_df):
    splits = make_cv_splits(synthetic_train_df, n_splits=5)
    summary = describe_splits(synthetic_train_df, splits)
    assert summary["n_val_rows"].sum() == len(synthetic_train_df)


def test_cv_reproducible(synthetic_train_df):
    """Same random_state → identical splits."""
    splits_a = make_cv_splits(synthetic_train_df, n_splits=5, random_state=42)
    splits_b = make_cv_splits(synthetic_train_df, n_splits=5, random_state=42)
    for (tr_a, val_a), (tr_b, val_b) in zip(splits_a, splits_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(val_a, val_b)


def test_pseudo_label_indices_never_in_val():
    """
    Pseudo-labeled rows appended after original data must never
    appear in any validation fold.
    """
    n_orig   = 80
    n_pseudo = 20

    # Simulate original splits on 80 rows
    base_df  = pd.DataFrame({
        "ID": [f"ID_TR_NEW_{i:04d}" for i in range(n_orig)],
        TARGET_COL: np.random.default_rng(0).integers(0, 2, n_orig),
    })
    splits = make_cv_splits(base_df, n_splits=5)

    # Extend train indices to include pseudo rows
    pseudo_indices = np.arange(n_orig, n_orig + n_pseudo)
    extended_splits = [
        (np.concatenate([tr, pseudo_indices]), val)
        for tr, val in splits
    ]

    # No pseudo index should appear in any val fold
    for fold, (tr, val) in enumerate(extended_splits):
        overlap = set(val) & set(pseudo_indices)
        assert len(overlap) == 0, (
            f"Fold {fold}: pseudo indices {overlap} appeared in validation"
        )


def test_iterative_pseudo_val_indices_clean():
    """
    Across all iterations, validation indices must only contain
    original training rows — never pseudo-labeled rows.
    """
    n_orig   = 80
    n_pseudo = 20

    base_df = pd.DataFrame({
        "ID": [f"ID_TR_NEW_{i:04d}" for i in range(n_orig)],
        TARGET_COL: np.random.default_rng(0).integers(0, 2, n_orig),
    })
    splits = make_cv_splits(base_df, n_splits=5)
    pseudo_indices = np.arange(n_orig, n_orig + n_pseudo)

    for iteration in range(3):
        # Simulate growing pseudo pool each iteration
        current_pseudo = pseudo_indices[:n_pseudo // (3 - iteration)]
        extended = [
            (np.concatenate([tr, current_pseudo]), val)
            for tr, val in splits
        ]
        for fold, (tr, val) in enumerate(extended):
            assert max(val) < n_orig, (
                f"Iter {iteration} Fold {fold}: val index {max(val)} "
                f">= n_orig {n_orig}"
            )


def test_get_single_window_indices():
    """get_single_window_indices must select exactly one window copy per base sample."""
    from pipelines.training.cv_strategy import get_single_window_indices

    # Create dummy data with 5 base samples, each repeated 3 times with _w0, _w1, _w2
    df = pd.DataFrame({
        "ID": [
            f"ID_TR_{i}_w{w}"
            for w in range(3)
            for i in range(5)
        ],
        TARGET_COL: [0, 1, 0, 1, 0] * 3
    })
    
    indices = get_single_window_indices(df, random_state=42)
    assert len(indices) == 5
    
    selected_ids = df.iloc[indices]["ID"].tolist()
    base_ids = [x.split("_w")[0] for x in selected_ids]
    assert len(set(base_ids)) == 5
    
    # Assert deterministic selections with random_state=42
    indices_2 = get_single_window_indices(df, random_state=42)
    assert np.array_equal(indices, indices_2)