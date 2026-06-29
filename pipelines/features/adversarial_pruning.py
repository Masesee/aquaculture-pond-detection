"""
Iterative Adversarial Feature Pruning.
Trains a model to distinguish train and test sets, and prunes the most
discriminative features until train and test distributions are aligned.

Outputs:
    outputs/features/invariant_features.txt   ← list of robust feature names
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

TARGET_AUC = 0.65  # Target threshold where distributions are considered aligned
MAX_PRUNED_PER_STEP = 5  # Number of features to prune per iteration
MAX_ITERATIONS = 25

def main() -> None:
    PROCESSED_DIR = ROOT / "data" / "processed"
    FEATURES_DIR = ROOT / "outputs" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
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

    # Initial feature set (exclude ID and label)
    feature_cols = [c for c in test_df.columns if c not in ["ID", TARGET_COL]]
    initial_count = len(feature_cols)
    
    # Prepare combined dataset
    X_train = train_df_single[feature_cols].copy()
    y_train = np.zeros(len(X_train))
    
    X_test = test_df[feature_cols].copy()
    y_test = np.ones(len(X_test))
    
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.concatenate([y_train, y_test])
    
    # Groups for Stratified Group CV
    train_groups = train_df_single["ID"].apply(lambda x: x.split("_w")[0]).values
    test_groups = test_df["ID"].values
    groups = np.concatenate([train_groups, test_groups])

    pruned_features = []
    
    print(f"=== Starting Iterative Adversarial Pruning (Target AUC < {TARGET_AUC}) ===")
    print(f"Initial feature count: {initial_count}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Update X to current feature subset
        X_curr = X[feature_cols]
        
        # 3-fold Stratified Group CV for speed and stability
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(X_curr))
        feature_importances = np.zeros(len(feature_cols))
        
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "max_depth": 4,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_curr, y, groups=groups)):
            X_tr, y_tr = X_curr.iloc[train_idx], y[train_idx]
            X_val, y_val = X_curr.iloc[val_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            feature_importances += model.feature_importances_ / 3.0
            
        auc = roc_auc_score(y, oof_probs)
        print(f"Iteration {iteration:02d} | Remaining Features: {len(feature_cols):3d} | Adversarial AUC: {auc:.4f}")
        
        # Stop if we are below target AUC or down to very few features
        if auc <= TARGET_AUC or len(feature_cols) <= 20:
            break
            
        # Identify features to prune
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances
        }).sort_values("importance", ascending=False)
        
        to_prune = importance_df.head(MAX_PRUNED_PER_STEP)["feature"].tolist()
        for f in to_prune:
            feature_cols.remove(f)
            pruned_features.append(f)
            
    print("\n=== Pruning Summary ===")
    print(f"Total features pruned: {len(pruned_features)} / {initial_count}")
    print(f"Remaining robust features: {len(feature_cols)}")
    
    # Save invariant features
    out_path = FEATURES_DIR / "invariant_features.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"Saved robust features to: {out_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
