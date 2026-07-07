import pytest
import pandas as pd
from pipelines.training.cv_strategy import (
    make_cv_splits,
    get_single_window_indices,
    get_fold_train_mask as shared_mask_builder
)
from pipelines.training.train import RANDOM_STATE, N_SPLITS
from pipelines.training.train import get_fold_train_mask as train_mask_builder
from pipelines.training.tune import get_fold_train_mask as tune_mask_builder

@pytest.fixture
def data_setup():
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])
    single_win_indices = get_single_window_indices(train_df, random_state=RANDOM_STATE)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)
    return train_df, base_ids_full, single_win_indices, train_df_single

def test_tune_and_train_produce_identical_fold_masks(data_setup):
    """
    Regression test for the Sub 62 capacity mismatch bug.
    Verifies that the fold train masks imported and used in both tune.py and train.py
    resolve to the exact same shared logic and yield identical row indices per fold,
    making silent divergence impossible.
    """
    train_df, base_ids_full, single_win_indices, train_df_single = data_setup
    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    # Verify they are physically the same function imported from cv_strategy
    assert train_mask_builder is shared_mask_builder, "train.py did not import the shared get_fold_train_mask function"
    assert tune_mask_builder is shared_mask_builder, "tune.py did not import the shared get_fold_train_mask function"

    mismatches = []
    for fold, (train_pos, val_pos) in enumerate(splits):
        train_mask = train_mask_builder(train_df, train_df_single, val_pos, base_ids_full)
        tune_mask = tune_mask_builder(train_df, train_df_single, val_pos, base_ids_full)

        train_idx = train_df.index[train_mask]
        tune_idx = train_df.index[tune_mask]

        if len(train_idx) != len(tune_idx) or not (train_idx == tune_idx).all():
            mismatches.append(
                f"Fold {fold}: train.py={len(train_idx)} rows, tune.py={len(tune_idx)} rows"
            )

    assert not mismatches, (
        "tune.py fold train selection diverges from train.py fold train selection:\n"
        + "\n".join(mismatches)
    )
