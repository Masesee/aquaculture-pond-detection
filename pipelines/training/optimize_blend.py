"""
Optimizes blend weights for LightGBM, XGBoost, and CatBoost models on raw OOF predictions.
Saves the resulting optimal weights to outputs/models/blend_weights.json.

Run with:
    python -m pipelines.training.optimize_blend
"""
import sys
from pathlib import Path
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.training.blend_config import save_blend_weights

MODELS_DIR = ROOT / "outputs" / "models"


def combined_score(f1: float, auc: float) -> float:
    return 0.6 * f1 + 0.4 * auc


def main() -> None:
    print("=== [Optimize Blend] Loading raw OOF predictions ===")
    
    lgbm_path = MODELS_DIR / "oof_predictions.csv"
    xgb_path = MODELS_DIR / "xgb_oof_predictions.csv"
    cb_path = MODELS_DIR / "cb_oof_predictions.csv"
    
    if not (lgbm_path.exists() and xgb_path.exists() and cb_path.exists()):
        print("ERROR: Missing OOF files. Run train.py, train_xgb.py, and train_catboost.py first.")
        sys.exit(1)
        
    lgbm_oof = pd.read_csv(lgbm_path)
    xgb_oof  = pd.read_csv(xgb_path)
    cb_oof   = pd.read_csv(cb_path)
    
    assert len(lgbm_oof) == len(xgb_oof) == len(cb_oof), "Mismatched OOF prediction lengths!"
    
    labels = lgbm_oof["label"].values
    p_lgb = lgbm_oof["oof_prob_raw"].values
    p_xgb = xgb_oof["oof_prob_raw"].values
    p_cb  = cb_oof["oof_prob_raw"].values
    
    def loss_func(weights):
        w1, w2, w3 = weights
        w_sum = w1 + w2 + w3
        if w_sum <= 0:
            return 10.0
        w1, w2, w3 = w1/w_sum, w2/w_sum, w3/w_sum
        
        p_blend = w1 * p_lgb + w2 * p_xgb + w3 * p_cb
        preds = (p_blend >= 0.5).astype(int)
        
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, p_blend)
        return -combined_score(f1, auc)

    print("  Running search for optimal weights using Differential Evolution...")
    res = differential_evolution(loss_func, bounds=[(0, 1), (0, 1), (0, 1)], seed=42)
    w_lgb, w_xgb, w_cb = res.x / res.x.sum()
    
    print(f"\n  Optimal Raw Weights: LGBM={w_lgb:.4f}, XGB={w_xgb:.4f}, CB={w_cb:.4f}")
    
    # Calculate scores
    p_blend_opt = w_lgb * p_lgb + w_xgb * p_xgb + w_cb * p_cb
    f1_opt = f1_score(labels, (p_blend_opt >= 0.5).astype(int))
    auc_opt = roc_auc_score(labels, p_blend_opt)
    score_opt = combined_score(f1_opt, auc_opt)
    
    p_equal = (p_lgb + p_xgb + p_cb) / 3.0
    f1_eq = f1_score(labels, (p_equal >= 0.5).astype(int))
    auc_eq = roc_auc_score(labels, p_equal)
    score_eq = combined_score(f1_eq, auc_eq)
    
    print(f"  Equal Weights OOF Score:   {score_eq:.4f} (F1={f1_eq:.4f}, AUC={auc_eq:.4f})")
    print(f"  Optimized Weights OOF Score: {score_opt:.4f} (F1={f1_opt:.4f}, AUC={auc_opt:.4f})")
    
    # Save the optimized weights
    weights = (float(w_lgb), float(w_xgb), float(w_cb))
    save_blend_weights(weights)
    print("  Saved optimal weights to outputs/models/blend_weights.json")


if __name__ == "__main__":
    main()
