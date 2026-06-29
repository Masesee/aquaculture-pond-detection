"""
Feature pipeline entrypoint.

Converts unobserved -9999 masked values to NaN, augments the training
dataset by simulating all 24 possible consecutive windows of length 4, 5,
and 6 months, and builds NaN-safe feature matrices.

Run with:
    python -m pipelines.features.build_features

Outputs:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    outputs/features/feature_pipeline_meta.json
    outputs/features/feature_names.txt
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from contracts.schema import DataSchema, TARGET_COL
from pipelines.features.aggregations import build_feature_matrix, feature_names

TRAIN_PATH     = ROOT / "data" / "raw" / "Train.csv"
TEST_PATH      = ROOT / "data" / "raw" / "Test.csv"
PROCESSED_DIR  = ROOT / "data" / "processed"
FEATURES_DIR   = ROOT / "outputs" / "features"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def augment_train_with_masks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments the training DataFrame by generating all 24 possible consecutive windows
    of length 4, 5, and 6 months, masking all other months to NaN.
    """
    from contracts.schema import MONTHS, ALL_BANDS
    augmented_dfs = []
    
    # 24 possible windows
    windows = []
    for length in [4, 5, 6]:
        for start in range(1, 13 - length + 2):
            windows.append((start, length))
            
    for idx, (start, length) in enumerate(windows):
        df_copy = df.copy()
        
        # Months to keep (1-indexed converted to string representation)
        keep_months = [f"{m:02d}" for m in range(start, start + length)]
        mask_months = [m for m in MONTHS if m not in keep_months]
        
        # Set all raw columns corresponding to masked months to np.nan
        for m in mask_months:
            for band in ALL_BANDS:
                col = f"{band}_{m}"
                if col in df_copy.columns:
                    df_copy[col] = np.nan
                    
        df_copy["ID"] = df_copy["ID"] + f"_w{idx}"
        augmented_dfs.append(df_copy)
        
    return pd.concat(augmented_dfs, ignore_index=True)


def main() -> None:
    print("=== Loading raw data ===")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    DataSchema.validate_train(train)
    DataSchema.validate_test(test)
    print(f"Train (raw): {train.shape} | Test (raw): {test.shape}")

    # Convert unobserved -9999 values to NaN immediately
    print("\n=== Replacing -9999 masks with NaN ===")
    train = train.replace(-9999, np.nan).replace(-9999.0, np.nan)
    test  = test.replace(-9999, np.nan).replace(-9999.0, np.nan)

    # Perform mask augmentation on training data
    print("\n=== Performing training data mask augmentation (24 windows per sample) ===")
    train_augmented = augment_train_with_masks(train)
    print(f"Train augmented shape: {train_augmented.shape}")

    # ── Build feature matrices ──
    print("\n=== Building feature matrices ===")
    train_feats = build_feature_matrix(train_augmented)
    test_feats  = build_feature_matrix(test)

    def apply_shap_filter(
        train_feats: pd.DataFrame,
        test_feats: pd.DataFrame,
        shap_path: Path,
        top_n: int = 80,
        always_keep: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters feature matrices to top_n features by mean |SHAP|.
        """
        if always_keep is None:
            always_keep = ["window_start", "window_length", "window_center"]

        importance = pd.read_csv(shap_path)
        existing_cols = set(train_feats.columns)
        valid_importance = importance[importance["feature"].isin(existing_cols)]
        
        if len(valid_importance) < len(importance):
            n_dropped = len(importance) - len(valid_importance)
            print(f"  WARNING: {n_dropped} features in SHAP importance not found in current matrix. "
                  "They will be ignored.")

        top_features = valid_importance.head(top_n)["feature"].tolist()
        keep = list(dict.fromkeys(top_features + always_keep))

        keep_train = ["ID", TARGET_COL] + [f for f in keep if f in train_feats.columns]
        keep_test  = ["ID"]             + [f for f in keep if f in test_feats.columns]

        print(f"  SHAP filter: keeping {len(keep)} features (from {train_feats.shape[1]-2} total)")
        return train_feats[keep_train], test_feats[keep_test]

    # Attach label to train features
    train_feats[TARGET_COL] = train_augmented[TARGET_COL].values

    # ── Optional: SHAP-based feature selection ─────────────────────────────
    shap_path = ROOT / "outputs" / "evaluation" / "shap_importance.csv"
    is_filtered = False

    if shap_path.exists():
        train_feats, test_feats = apply_shap_filter(
            train_feats, test_feats, shap_path, top_n=80
        )
        is_filtered = True
    else:
        print("  No SHAP importance file found — using all features")

    print(f"  Train features shape: {train_feats.shape}")
    print(f"  Test  features shape: {test_feats.shape}")

    # Validate feature count matches contract
    if not is_filtered:
        expected = feature_names(exclude_id=True)
        actual   = [c for c in train_feats.columns if c not in ["ID", TARGET_COL]]
        assert actual == expected, (
            f"Feature column mismatch.\n"
            f"Expected {len(expected)}, got {len(actual)}\n"
            f"Missing: {set(expected) - set(actual)}\n"
            f"Extra:   {set(actual) - set(expected)}"
        )
        print(f"  Feature contract validated: {len(expected)} features")
    else:
        print("  Feature contract skipped (SHAP filter applied)")

    # ── Save ──
    print("\n=== Saving ===")
    train_feats.to_parquet(PROCESSED_DIR / "train_features.parquet", index=False)
    test_feats.to_parquet(PROCESSED_DIR  / "test_features.parquet",  index=False)
    print("  Saved: data/processed/train_features.parquet")
    print("  Saved: data/processed/test_features.parquet")

    # Save metadata
    final_features = [c for c in train_feats.columns if c not in ["ID", TARGET_COL]]
    meta = {
        "n_features": len(final_features),
        "feature_names": final_features,
        "is_shap_filtered": is_filtered,
    }
    with open(FEATURES_DIR / "feature_pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save ordered feature names as plain text for easy inspection
    with open(FEATURES_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(final_features))

    print("  Saved: outputs/features/feature_pipeline_meta.json")
    print("  Saved: outputs/features/feature_names.txt")
    print("\n=== Feature pipeline complete ===")


if __name__ == "__main__":
    main()
