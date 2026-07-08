import sys
import time
from pathlib import Path
import pandas as pd
import xgboost as xgb
import catboost as cb

ROOT = Path("D:/Data Science/GeoAI Aquaculture Pond Identification/aquaculture-pond-detection")
sys.path.insert(0, str(ROOT))

from contracts.schema import TARGET_COL, WINDOW_METADATA_COLS
from pipelines.training.cv_strategy import make_cv_splits, get_single_window_indices, get_fold_train_mask

PROCESSED_DIR = ROOT / "data/processed"

def benchmark_models():
    print("=== Loading features for benchmark ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    
    invariant_path = ROOT / "outputs/features/invariant_features.txt"
    with open(invariant_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
        
    for col in WINDOW_METADATA_COLS:
        if col not in feature_cols and col in train_df.columns:
            feature_cols.append(col)
            
    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])
    y_train_single = train_df_single[TARGET_COL].values
    
    splits = make_cv_splits(train_df_single, n_splits=5, random_state=42)
    
    # 1. XGBoost Benchmark
    print("\n=== Benchmarking XGBoost (1 Trial, 5 Folds) ===")
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    }
    
    start = time.time()
    for fold, (train_pos, val_pos) in enumerate(splits):
        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values
        X_val = train_df_single[feature_cols].iloc[val_pos]
        y_val = y_train_single[val_pos]
        
        model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        _ = model.predict_proba(X_val)[:, 1]
    xgb_duration = time.time() - start
    print(f"XGBoost 5-fold CV took: {xgb_duration:.2f} seconds")
    
    # 2. CatBoost Benchmark
    print("\n=== Benchmarking CatBoost (1 Trial, 5 Folds) ===")
    cb_params = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }
    
    start = time.time()
    for fold, (train_pos, val_pos) in enumerate(splits):
        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values
        X_val = train_df_single[feature_cols].iloc[val_pos]
        y_val = y_train_single[val_pos]
        
        model = cb.CatBoostClassifier(**cb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        _ = model.predict_proba(X_val)[:, 1]
    cb_duration = time.time() - start
    print(f"CatBoost 5-fold CV took: {cb_duration:.2f} seconds")

if __name__ == "__main__":
    benchmark_models()
