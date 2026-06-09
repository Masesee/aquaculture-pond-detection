"""
Cross-validation strategy for aquaculture pond detection.

Stratifies on the interaction of label × region to ensure every fold
preserves both class balance and regional distribution.

Stratum encoding:
  0 = non-pond, region 0
  1 = non-pond, region 1
  2 = pond,     region 0
  3 = pond,     region 1
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from contracts.schema import TARGET_COL


def make_strata(labels: pd.Series, regions: pd.Series) -> pd.Series:
    """
    Encodes label × region interaction as a single integer stratum.

    Parameters
    ----------
    labels  : binary 0/1 Series
    regions : binary 0/1 Series, same index

    Returns
    -------
    pd.Series of int in {0, 1, 2, 3}
    """
    assert labels.index.equals(regions.index), "labels and regions must share the same index"
    return (labels * 2 + regions).astype(int)


def make_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (train_idx, val_idx) integer position pairs
    for stratified k-fold CV on label × region strata.

    Parameters
    ----------
    df           : feature DataFrame containing TARGET_COL and 'region'
    n_splits     : number of folds
    random_state : for reproducibility

    Returns
    -------
    List of (train_positions, val_positions) as numpy integer arrays.
    These are positional indices into df, not index labels.
    """
    strata = make_strata(df[TARGET_COL], df["region"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(df)), strata.values))


def describe_splits(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing class and region balance per fold.
    Use this to verify splits before training.
    """
    rows = []
    for fold_idx, (train_pos, val_pos) in enumerate(splits):
        val_df = df.iloc[val_pos]
        rows.append({
            "fold":           fold_idx,
            "n_val":          len(val_pos),
            "val_pond_rate":  val_df[TARGET_COL].mean().round(4),
            "val_region0_n":  (val_df["region"] == 0).sum(),
            "val_region1_n":  (val_df["region"] == 1).sum(),
            "val_pond_r0":    val_df.loc[val_df["region"] == 0, TARGET_COL].mean().round(4),
            "val_pond_r1":    val_df.loc[val_df["region"] == 1, TARGET_COL].mean().round(4),
        })
    return pd.DataFrame(rows)