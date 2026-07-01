"""
Blended ensemble pipeline.

Blends probabilities from LightGBM and XGBoost models (50/50 weighting),
evaluates blended OOF metrics, and builds the final submission file.

Run with:
    python -m pipelines.training.blend_ensemble
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import joblib
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL
from pipelines.training.train import correct_prior

MODELS_DIR     = ROOT / "outputs" / "models"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"
PROCESSED_DIR  = ROOT / "data" / "processed"


def combined_score(f1: float, auc: float) -> float:
    return 0.6 * f1 + 0.4 * auc


def main() -> None:
    test_prior = 0.55
    if "--test-prior" in sys.argv:
        try:
            idx = sys.argv.index("--test-prior") + 1
            test_prior = float(sys.argv[idx])
        except (ValueError, IndexError):
            pass

    print("=== [Ensemble] Loading OOF predictions ===")
    lgbm_oof = pd.read_csv(MODELS_DIR / "oof_predictions.csv")
    xgb_oof  = pd.read_csv(MODELS_DIR / "xgb_oof_predictions.csv")
    cb_oof   = pd.read_csv(MODELS_DIR / "cb_oof_predictions.csv")

    assert len(lgbm_oof) == len(xgb_oof) == len(cb_oof), "Mismatched OOF rows between models!"

    labels = lgbm_oof["label"].values
    lgbm_probs = lgbm_oof["oof_prob_cal"].values
    xgb_probs  = xgb_oof["oof_prob_cal"].values
    cb_probs   = cb_oof["oof_prob_cal"].values

    # Equal 1/3 weighted triad ensemble OOF probabilities
    blend_oof_probs = (lgbm_probs + xgb_probs + cb_probs) / 3.0
    blend_preds = (blend_oof_probs >= 0.5).astype(int)

    f1  = f1_score(labels, blend_preds)
    auc = roc_auc_score(labels, blend_oof_probs)
    score = combined_score(f1, auc)

    print(f"  LGBM  OOF — F1={f1_score(labels, (lgbm_probs>=0.5).astype(int)):.4f} | AUC={roc_auc_score(labels, lgbm_probs):.4f}")
    print(f"  XGB   OOF — F1={f1_score(labels, (xgb_probs>=0.5).astype(int)):.4f} | AUC={roc_auc_score(labels, xgb_probs):.4f}")
    print(f"  CB    OOF — F1={f1_score(labels, (cb_probs>=0.5).astype(int)):.4f} | AUC={roc_auc_score(labels, cb_probs):.4f}")
    print(f"  TRIAD OOF — F1={f1:.4f} | AUC={auc:.4f} | Score={score:.4f}")

    # Load GRU OOF predictions if available
    gru_path = MODELS_DIR / "gru_oof_probs.csv"
    gru_test_path = MODELS_DIR / "gru_test_probs.csv"
    
    import numpy as np
    w_gru = 0.0
    gru_oof_probs_aligned = np.zeros_like(blend_oof_probs)
    gru_test_probs_aligned = np.zeros(1030) # test size
    
    if "--use-gru" in sys.argv and gru_path.exists() and gru_test_path.exists():
        gru_oof = pd.read_csv(gru_path)
        gru_test = pd.read_csv(gru_test_path)
        
        # Align GRU OOF by ID
        gru_id_to_prob = dict(zip(gru_oof["ID"], gru_oof["gru_prob"]))
        gru_oof_probs_aligned = np.array([gru_id_to_prob[idx] for idx in lgbm_oof["ID"]])
        
        print("  Loaded GRU model predictions. Optimizing blend weights...")
        
        best_w = 0.0
        best_score = score
        best_f1 = f1
        best_auc = auc
        
        for w in np.linspace(0.0, 0.40, 41):
            temp_blend = (1.0 - w) * blend_oof_probs + w * gru_oof_probs_aligned
            temp_preds = (temp_blend >= 0.5).astype(int)
            temp_f1 = f1_score(labels, temp_preds)
            temp_auc = roc_auc_score(labels, temp_blend)
            temp_score = combined_score(temp_f1, temp_auc)
            if temp_score > best_score:
                best_score = temp_score
                best_w = w
                best_f1 = temp_f1
                best_auc = temp_auc
                
        w_gru = best_w
        print(f"  Optimal GRU weight: {w_gru:.2f} (Triad weight: {1.0 - w_gru:.2f})")
        print(f"  Blended OOF — F1={best_f1:.4f} | AUC={best_auc:.4f} | Score={best_score:.4f}")
        
        # Update OOF probabilities for saving
        blend_oof_probs = (1.0 - w_gru) * blend_oof_probs + w_gru * gru_oof_probs_aligned

    # Save ensemble OOF summary
    ensemble_oof_df = pd.DataFrame({
        "ID": lgbm_oof["ID"],
        "label": labels,
        "lgbm_prob": lgbm_probs,
        "xgb_prob": xgb_probs,
        "cb_prob": cb_probs,
        "blend_prob": blend_oof_probs,
    })
    ensemble_oof_df.to_csv(MODELS_DIR / "ensemble_oof_predictions.csv", index=False)

    print("\n=== [Ensemble] Building final blended test predictions ===")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")
    xgb_test_df = pd.read_csv(MODELS_DIR / "xgb_test_probs.csv")
    cb_test_df  = pd.read_csv(MODELS_DIR / "cb_test_probs.csv")

    # Load final models and calibrators for test inference
    lgbm_model = joblib.load(MODELS_DIR / "lgbm_model.joblib")
    lgbm_cal   = joblib.load(MODELS_DIR / "calibrator.joblib")
    
    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    with open(invariant_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    exclude_metadata = "--exclude-metadata" in sys.argv
    metadata_cols = [
        "window_start", "window_length", "window_center",
        "window_start_sin", "window_start_cos",
        "window_center_sin", "window_center_cos"
    ]
    if exclude_metadata:
        feature_cols = [c for c in feature_cols if c not in metadata_cols]
    else:
        for col in metadata_cols:
            if col not in feature_cols and col in test_df.columns:
                feature_cols.append(col)


    X_test = test_df[feature_cols]
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    train_prior = train_df[TARGET_COL].mean()

    # LGBM calibrated test probs
    raw_lgbm_test = lgbm_model.predict_proba(X_test)[:, 1]
    cal_lgbm_test = lgbm_cal.transform(raw_lgbm_test)

    # XGB and CB calibrated test probs
    xgb_test_probs = xgb_test_df["xgb_prob_cal"].values
    cb_test_probs  = cb_test_df["cb_prob_cal"].values

    # Blend test probabilities
    triad_test_probs = (cal_lgbm_test + xgb_test_probs + cb_test_probs) / 3.0
    
    if w_gru > 0.0:
        # Align GRU Test by ID
        gru_test = pd.read_csv(gru_test_path)
        gru_test_id_to_prob = dict(zip(gru_test["ID"], gru_test["gru_prob"]))
        gru_test_probs_aligned = np.array([gru_test_id_to_prob[idx] for idx in test_df["ID"]])
        blend_test_probs = (1.0 - w_gru) * triad_test_probs + w_gru * gru_test_probs_aligned
    else:
        blend_test_probs = triad_test_probs

    blend_test_corrected = correct_prior(blend_test_probs, train_prior, test_prior)
    binary_preds = (blend_test_corrected >= 0.5).astype(int)

    print(f"  Test predicted positive rate (corrected): {binary_preds.mean():.3f}")
    print(f"  Predicted ponds: {binary_preds.sum()} / {len(binary_preds)}")

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "TargetF1": binary_preds,
        "TargetRAUC": blend_test_corrected,
    })
    submission_path = SUBMISSIONS_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"  Saved final blended submission: {submission_path}")


if __name__ == "__main__":
    main()
