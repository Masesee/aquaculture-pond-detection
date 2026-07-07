"""
Cross-validation strategy for aquaculture pond detection.

Stratifies on label and groups by the base ID (before mask augmentation suffix)
using StratifiedGroupKFold to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from contracts.schema import TARGET_COL


def make_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (train_idx, val_idx) position pairs using StratifiedGroupKFold
    grouped by original sample ID.

    Parameters
    ----------
    df           : feature DataFrame containing TARGET_COL and 'ID'
    n_splits     : number of folds
    random_state : for reproducibility

    Returns
    -------
    List of (train_positions, val_positions) as numpy integer arrays.
    """
    # Extract base IDs as groups to keep augmented versions of the same sample together
    groups = df["ID"].apply(lambda x: x.split("_w")[0]).values
    labels = df[TARGET_COL].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(sgkf.split(df, labels, groups=groups))


def describe_splits(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing class balance and group distribution per fold.
    """
    rows = []
    groups = df["ID"].apply(lambda x: x.split("_w")[0])
    for fold_idx, (train_pos, val_pos) in enumerate(splits):
        val_df = df.iloc[val_pos]
        val_groups = groups.iloc[val_pos]
        rows.append({
            "fold":           fold_idx,
            "n_val_rows":     len(val_pos),
            "n_val_groups":   val_groups.nunique(),
            "val_pond_rate":  val_df[TARGET_COL].mean().round(4),
        })
    return pd.DataFrame(rows)


def get_single_window_indices(df: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    """
    Returns a numpy array of integer indices representing exactly one window
    version per unique base ID in df.
    """
    base_ids = df["ID"].apply(lambda x: x.split("_w")[0])
    grouped = df.groupby(base_ids).indices
    
    selected_indices = []
    rng = np.random.default_rng(random_state)
    for base, idxs in sorted(grouped.items()):
        selected_idx = rng.choice(idxs)
        selected_indices.append(selected_idx)
        
    return np.array(selected_indices)


def get_val_base_ids(df_single: pd.DataFrame, val_pos: np.ndarray) -> set[str]:
    """
    Extracts the set of base IDs for the validation fold from the single-window dataset.
    """
    return set(df_single.iloc[val_pos]["ID"].apply(lambda x: x.split("_w")[0]))


def get_train_mask_for_val_base_ids(
    base_ids_full: pd.Series,
    val_base_ids: set[str],
) -> np.ndarray:
    """
    Returns a boolean mask of train_df rows whose base ID is NOT in val_base_ids.
    Used to build the fold training set from the augmented dataset without leakage.
    """
    return (~base_ids_full.isin(val_base_ids)).values


def get_fold_train_mask(
    train_df: pd.DataFrame,
    train_df_single: pd.DataFrame,
    val_pos: np.ndarray,
    base_ids_full: pd.Series,
) -> np.ndarray:
    """
    Returns the training mask for a given fold validation set, mapping back to train_df.
    """
    val_base_ids = get_val_base_ids(train_df_single, val_pos)
    return get_train_mask_for_val_base_ids(base_ids_full, val_base_ids)
