"""
CatBoost training pipeline.

Trains a CatBoost Classifier on the domain-invariant feature set using single-window
Stratified Group CV, saving OOF predictions and test probabilities for ensembling.

Run with:
    python -m pipelines.training.train_catboost
"""

import sys
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL
from pipelines.training.cv_strategy import (
    make_cv_splits,
    get_single_window_indices,
)
from pipelines.training.calibration import (
    fit_calibrator,
    apply_calibrator,
)

PROCESSED_DIR  = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

CAT_PARAMS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 0,
}


def combined_score(f1: float, auc: float) -> float:
    return 0.6 * f1 + 0.4 * auc


def correct_prior(probs: np.ndarray, train_prior: float, test_prior: float) -> np.ndarray:
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0 - eps)
    ratio_pos = test_prior / train_prior
    ratio_neg = (1.0 - test_prior) / (1.0 - train_prior)
    corrected = (probs * ratio_pos) / (probs * ratio_pos + (1.0 - probs) * ratio_neg)
    return np.clip(corrected, 0.0, 1.0)


def main() -> None:
    print("=== [CatBoost] Loading feature matrices ===")

    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    if invariant_path.exists():
        with open(invariant_path) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(feature_cols)} robust features from invariant_features.txt")
    else:
        feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]

    metadata_cols = [
        "window_start", "window_length", "window_center",
        "window_start_sin", "window_start_cos",
        "window_center_sin", "window_center_cos"
    ]
    feature_cols = [c for c in feature_cols if c not in metadata_cols]
    print(f"  Excluded window metadata. Remaining: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]

    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)

    print(f"\n=== [CatBoost] {N_SPLITS}-fold stratified group CV (on 1-window subset) ===")
    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    oof_probs   = np.zeros(len(train_df_single), dtype=float)
    fold_scores = []

    for fold, (train_pos, val_pos) in enumerate(splits):
        val_idx_in_train_df = single_win_indices[val_pos]
        val_base_ids = set(train_df_single.iloc[val_pos]["ID"].apply(lambda x: x.split("_w")[0]))

        X_val = train_df.iloc[val_idx_in_train_df][feature_cols]
        y_val = train_df.iloc[val_idx_in_train_df][TARGET_COL].values

        train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values

        model = cb.CatBoostClassifier(**CAT_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )

        fold_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_pos] = fold_probs

        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_f1    = f1_score(y_val, fold_preds)
        fold_auc   = roc_auc_score(y_val, fold_probs)
        fold_score = combined_score(fold_f1, fold_auc)

        fold_scores.append({"fold": fold, "f1": round(fold_f1, 4), "auc": round(fold_auc, 4), "score": round(fold_score, 4)})
        print(f"  Fold {fold}: F1={fold_f1:.4f} | AUC={fold_auc:.4f} | Score={fold_score:.4f}")

    y_train_single = train_df_single[TARGET_COL].values
    oof_preds = (oof_probs >= 0.5).astype(int)
    oof_f1    = f1_score(y_train_single, oof_preds)
    oof_auc   = roc_auc_score(y_train_single, oof_probs)
    oof_score = combined_score(oof_f1, oof_auc)
    print(f"\n  [CatBoost] OOF aggregate (pre-cal) — F1={oof_f1:.4f} | AUC={oof_auc:.4f} | Score={oof_score:.4f}")

    print("\n=== [CatBoost] Fitting calibrator on OOF predictions ===")
    calibrator    = fit_calibrator(oof_probs, y_train_single)
    cal_probs_oof = apply_calibrator(calibrator, oof_probs)
    cal_preds_oof = (cal_probs_oof >= 0.5).astype(int)
    cal_f1        = f1_score(y_train_single, cal_preds_oof)
    cal_auc       = roc_auc_score(y_train_single, cal_probs_oof)
    cal_score     = combined_score(cal_f1, cal_auc)
    print(f"  [CatBoost] Calibrated OOF — F1={cal_f1:.4f} | AUC={cal_auc:.4f} | Score={cal_score:.4f}")

    print("\n=== [CatBoost] Training final model on full training data ===")
    final_model = cb.CatBoostClassifier(**CAT_PARAMS)
    final_model.fit(X_train, y_train)

    print("\n=== [CatBoost] Generating test predictions ===")
    raw_test_probs  = final_model.predict_proba(X_test)[:, 1]
    cal_test_probs  = apply_calibrator(calibrator, raw_test_probs)


    oof_df = pd.DataFrame({
        "ID": train_df_single["ID"].values,
        "label": y_train_single,
        "oof_prob_raw": oof_probs,
        "oof_prob_cal": cal_probs_oof,
    })
    oof_df.to_csv(MODELS_DIR / "cb_oof_predictions.csv", index=False)

    test_probs_df = pd.DataFrame({
        "ID": test_df["ID"].values,
        "cb_prob_raw": raw_test_probs,
        "cb_prob_cal": cal_test_probs,
    })
    test_probs_df.to_csv(MODELS_DIR / "cb_test_probs.csv", index=False)


    joblib.dump(final_model, MODELS_DIR / "cb_model.joblib")
    joblib.dump(calibrator, MODELS_DIR / "cb_calibrator.joblib")
    print("  Saved: outputs/models/cb_model.joblib")
    print("  Saved: outputs/models/cb_oof_predictions.csv")
    print("  Saved: outputs/models/cb_test_probs.csv")


if __name__ == "__main__":
    main()
