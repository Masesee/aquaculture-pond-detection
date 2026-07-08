import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("D:/Data Science/GeoAI Aquaculture Pond Identification/aquaculture-pond-detection")
sys.path.insert(0, str(ROOT))

from contracts.schema import TARGET_COL, WINDOW_METADATA_COLS
from pipelines.training.cv_strategy import make_cv_splits, get_single_window_indices, get_fold_train_mask
from pipelines.training.blending import blend_raw_probs
from pipelines.evaluation.metrics import combined_score

PROCESSED_DIR = ROOT / "data/processed"
MODELS_DIR = ROOT / "outputs/models"

def load_features():
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    invariant_path = ROOT / "outputs/features/invariant_features.txt"
    with open(invariant_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
        
    for col in WINDOW_METADATA_COLS:
        if col not in feature_cols and col in train_df.columns:
            feature_cols.append(col)
            
    return train_df, feature_cols

def optimize_weights(lgbm_oof, xgb_oof, cb_oof, y_true):
    def loss_fn(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        p_blend = blend_raw_probs(lgbm_oof, xgb_oof, cb_oof, w)
        preds = (p_blend >= 0.5).astype(int)
        
        f1 = f1_score(y_true, preds)
        auc = roc_auc_score(y_true, p_blend)
        score = combined_score(f1, auc)
        return -score  # Minimize negative score
        
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    res = differential_evolution(loss_fn, bounds, seed=42)
    best_w = res.x / np.sum(res.x)
    return best_w, -res.fun

def run_seed_check(seed):
    print(f"\n--- Running Training and Optimization for Seed {seed} ---")
    train_df, feature_cols = load_features()
    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])
    
    # 1. Load Tuned Params
    with open(MODELS_DIR / "best_params.json") as f:
        lgb_params = json.load(f)
    lgb_params["random_state"] = seed
    lgb_params["verbose"] = -1
    
    with open(MODELS_DIR / "best_params_xgb.json") as f:
        xgb_params = json.load(f)
    xgb_params["random_state"] = seed
    xgb_params["n_jobs"] = -1
    
    with open(MODELS_DIR / "best_params_cb.json") as f:
        cb_params = json.load(f)
    cb_params["random_seed"] = seed
    cb_params["verbose"] = 0
    cb_params["thread_count"] = -1
    
    splits = make_cv_splits(train_df_single, n_splits=5, random_state=42)
    y_true = train_df_single[TARGET_COL].values
    
    lgb_oof = np.zeros(len(y_true))
    xgb_oof = np.zeros(len(y_true))
    cb_oof = np.zeros(len(y_true))
    
    for fold, (train_pos, val_pos) in enumerate(splits):
        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values
        X_val = train_df_single[feature_cols].iloc[val_pos]
        y_val = y_true[val_pos]
        
        # LGBM
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr)
        lgb_oof[val_pos] = model_lgb.predict_proba(X_val)[:, 1]
        
        # XGBoost (with constructor early stopping)
        model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_oof[val_pos] = model_xgb.predict_proba(X_val)[:, 1]
        
        # CatBoost (with fit-level early stopping)
        model_cb = cb.CatBoostClassifier(**cb_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        cb_oof[val_pos] = model_cb.predict_proba(X_val)[:, 1]
        
    # Check individual model scores
    lgb_score = combined_score(f1_score(y_true, (lgb_oof >= 0.5).astype(int)), roc_auc_score(y_true, lgb_oof))
    xgb_score = combined_score(f1_score(y_true, (xgb_oof >= 0.5).astype(int)), roc_auc_score(y_true, xgb_oof))
    cb_score = combined_score(f1_score(y_true, (cb_oof >= 0.5).astype(int)), roc_auc_score(y_true, cb_oof))
    
    print(f"  [OOF Score] LGBM:     {lgb_score:.6f}")
    print(f"  [OOF Score] XGBoost:  {xgb_score:.6f}")
    print(f"  [OOF Score] CatBoost: {cb_score:.6f}")
    
    # Optimize blend weights
    best_w, opt_score = optimize_weights(lgb_oof, xgb_oof, cb_oof, y_true)
    print(f"  Optimal Weights: LGBM={best_w[0]:.4f}, XGB={best_w[1]:.4f}, CB={best_w[2]:.4f}")
    print(f"  Optimized Blend OOF Score: {opt_score:.6f}")
    return best_w, opt_score

if __name__ == "__main__":
    seeds = [42, 100, 2026, 77, 999, 1337, 8888]
    results = {}
    for s in seeds:
        w, score = run_seed_check(s)
        results[s] = {"weights": w, "score": score}
        
    print("\n=== Summary Table ===")
    print(f"{'Seed':<6} | {'LGBM':<6} | {'XGB':<6} | {'CB':<6} | {'OOF Score':<10}")
    print("-" * 46)
    for s in seeds:
        w = results[s]["weights"]
        sc = results[s]["score"]
        print(f"{s:<6} | {w[0]:.4f} | {w[1]:.4f} | {w[2]:.4f} | {sc:.6f}")
