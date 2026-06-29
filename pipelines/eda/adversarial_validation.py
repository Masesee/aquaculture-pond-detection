"""
Adversarial validation script.
Identifies features that differ most between the training set (augmented)
and the test set by training a model to distinguish between them.

Run with:
    python -m pipelines.eda.adversarial_validation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contracts.schema import TARGET_COL

def main() -> None:
    PROCESSED_DIR = ROOT / "data" / "processed"
    
    train_path = PROCESSED_DIR / "train_features.parquet"
    test_path = PROCESSED_DIR / "test_features.parquet"
    
    if not train_path.exists() or not test_path.exists():
        print("ERROR: Processed feature matrices not found. Please run feature builder first.")
        sys.exit(1)
        
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Use 1-window-per-sample subset for train set to match test set distribution format
    from pipelines.training.cv_strategy import get_single_window_indices
    single_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_indices].reset_index(drop=True)

    # Exclude ID and target
    feature_cols = [c for c in test_df.columns if c not in ["ID", TARGET_COL]]
    
    print(f"Loaded {len(train_df_single)} train rows (1 window per sample) and {len(test_df)} test rows.")
    print(f"Evaluating {len(feature_cols)} features...")

    # Create adversarial validation dataset
    X_train = train_df_single[feature_cols].copy()
    y_train = np.zeros(len(X_train))
    
    X_test = test_df[feature_cols].copy()
    y_test = np.ones(len(X_test))
    
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.concatenate([y_train, y_test])
    
    # Set up groups to prevent leakage of train window duplicates
    train_groups = train_df_single["ID"].apply(lambda x: x.split("_w")[0]).values
    test_groups = test_df["ID"].values
    groups = np.concatenate([train_groups, test_groups])

    # Run 5-fold Stratified Group CV
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))
    
    feature_importances = np.zeros(len(feature_cols))
    
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    print("\n=== Training Adversarial Classifier ===")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        feature_importances += model.feature_importances_ / 5.0
        
    auc = roc_auc_score(y, oof_probs)
    print(f"\nAdversarial OOF ROC-AUC: {auc:.4f}")
    
    if auc > 0.55:
        print("\nWARNING: Train and test distributions are significantly different!")
        print("Here are the top 20 features causing the shift:")
        
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        
        print(importance_df.head(20).to_string())
    else:
        print("\nSUCCESS: Train and test distributions are well aligned (ROC-AUC ~ 0.50).")

if __name__ == "__main__":
    main()
