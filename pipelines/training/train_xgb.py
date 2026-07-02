"""
XGBoost training pipeline.

Trains an XGBoost model on the domain-invariant feature set using single-window
Stratified Group CV, saving OOF predictions and test probabilities for ensembling.

Run with:
    python -m pipelines.training.train_xgb
"""

import sys
import json
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL, WINDOW_METADATA_COLS
from pipelines.training.cv_strategy import (
    make_cv_splits,
    get_single_window_indices,
)

N_SPLITS = 5
RANDOM_STATE = 42

def combined_score(f1: float, auc: float) -> float:
    return 0.6 * f1 + 0.4 * auc

from pipelines.training.calibration import (
    fit_calibrator,
    apply_calibrator,
)

PROCESSED_DIR  = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "outputs" / "models"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.4967,
    "reg_alpha": 0.183,
    "reg_lambda": 2.70,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def correct_prior(probs: np.ndarray, train_prior: float, test_prior: float) -> np.ndarray:
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0 - eps)
    ratio_pos = test_prior / train_prior
    ratio_neg = (1.0 - test_prior) / (1.0 - train_prior)
    corrected = (probs * ratio_pos) / (probs * ratio_pos + (1.0 - probs) * ratio_neg)
    return np.clip(corrected, 0.0, 1.0)


def main() -> None:
    print("=== [XGBoost] Loading feature matrices ===")

    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    if invariant_path.exists():
        with open(invariant_path) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(feature_cols)} robust features from invariant_features.txt")
    else:
        feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]

    exclude_metadata = "--exclude-metadata" in sys.argv
    metadata_cols = WINDOW_METADATA_COLS
    if exclude_metadata:
        feature_cols = [c for c in feature_cols if c not in metadata_cols]
        print(f"  Excluded window metadata. Remaining: {len(feature_cols)}")
    else:
        for col in metadata_cols:
            if col not in feature_cols and col in train_df.columns:
                feature_cols.append(col)
        print(f"  Included window metadata. Total features: {len(feature_cols)}")


    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]

    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)
    y_train_single = train_df_single[TARGET_COL].values

    # ── Optional pseudo-labeling ───────────────────────────────────────────
    use_pseudo = "--pseudo" in sys.argv
    if use_pseudo:
        from pipelines.training.train import load_pseudo_labels
        sub_path = SUBMISSIONS_DIR / "blend_submission_best.csv"
        if not sub_path.exists():
            sub_path = SUBMISSIONS_DIR / "submission.csv"
        if not sub_path.exists():
            print(f"\nERROR: Pseudo-labeling requires a source submission at: {sub_path}")
            return
        print("\n=== [XGBoost] Loading pseudo-labels ===")
        pseudo_df, n_pseudo = load_pseudo_labels(
            submission_path    = sub_path,
            test_features_path = PROCESSED_DIR   / "test_features.parquet",
        )
        train_augmented = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"  Training size: {len(train_df)} -> {len(train_augmented)} (+{n_pseudo} pseudo-labeled)")
    else:
        train_augmented = train_df

    # Load tuned parameters if they exist
    best_params_path = MODELS_DIR / "best_params_xgb.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            _bp = json.load(f)
        active_params = {**XGB_PARAMS, **_bp}
        print("  Loaded tuned params from best_params_xgb.json")
    else:
        active_params = XGB_PARAMS

    print(f"\n=== [XGBoost] {N_SPLITS}-fold stratified group CV (on 1-window subset) ===")
    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    oof_probs   = np.zeros(len(train_df_single), dtype=float)
    fold_scores = []

    for fold, (train_pos, val_pos) in enumerate(splits):
        val_base_ids = set(train_df_single.iloc[val_pos]["ID"].apply(lambda x: x.split("_w")[0]))

        if use_pseudo:
            train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
            pseudo_mask = np.ones(len(pseudo_df), dtype=bool)
            full_train_mask = np.concatenate([train_mask, pseudo_mask])
            X_tr = train_augmented.loc[full_train_mask, feature_cols]
            y_tr = train_augmented.loc[full_train_mask, TARGET_COL].values
        else:
            train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
            X_tr = train_df.loc[train_mask, feature_cols]
            y_tr = train_df.loc[train_mask, TARGET_COL].values

        X_val = train_df_single[feature_cols].iloc[val_pos]
        y_val = y_train_single[val_pos]

        model = xgb.XGBClassifier(**active_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
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

    oof_preds = (oof_probs >= 0.5).astype(int)
    oof_f1    = f1_score(y_train_single, oof_preds)
    oof_auc   = roc_auc_score(y_train_single, oof_probs)
    oof_score = combined_score(oof_f1, oof_auc)
    print(f"\n  [XGBoost] OOF aggregate (pre-cal) — F1={oof_f1:.4f} | AUC={oof_auc:.4f} | Score={oof_score:.4f}")

    print("\n=== [XGBoost] Fitting calibrator on OOF predictions ===")
    calibrator    = fit_calibrator(oof_probs, y_train_single)
    cal_probs_oof = apply_calibrator(calibrator, oof_probs)
    cal_preds_oof = (cal_probs_oof >= 0.5).astype(int)
    cal_f1        = f1_score(y_train_single, cal_preds_oof)
    cal_auc       = roc_auc_score(y_train_single, cal_probs_oof)
    cal_score     = combined_score(cal_f1, cal_auc)
    print(f"  [XGBoost] Calibrated OOF — F1={cal_f1:.4f} | AUC={cal_auc:.4f} | Score={cal_score:.4f}")

    print("\n=== [XGBoost] Training final model on full training data (Seed Averaging across [42, 100, 2026]) ===")
    seeds = [42, 100, 2026]
    raw_test_probs_list = []
    
    for seed in seeds:
        seed_params = {**active_params, "random_state": seed}
        model = xgb.XGBClassifier(**seed_params)
        if use_pseudo:
            model.fit(train_augmented[feature_cols], train_augmented[TARGET_COL].values)
        else:
            model.fit(X_train, y_train)
        raw_test_probs_list.append(model.predict_proba(X_test)[:, 1])
        if seed == 42:
            joblib.dump(model, MODELS_DIR / "xgb_model.joblib")
            
    raw_test_probs = np.mean(raw_test_probs_list, axis=0)
    cal_test_probs  = apply_calibrator(calibrator, raw_test_probs)


    oof_df = pd.DataFrame({
        "ID": train_df_single["ID"].values,
        "label": y_train_single,
        "oof_prob_raw": oof_probs,
        "oof_prob_cal": cal_probs_oof,
    })
    oof_df.to_csv(MODELS_DIR / "xgb_oof_predictions.csv", index=False)

    test_probs_df = pd.DataFrame({
        "ID": test_df["ID"].values,
        "xgb_prob_raw": raw_test_probs,
        "xgb_prob_cal": cal_test_probs,
    })
    test_probs_df.to_csv(MODELS_DIR / "xgb_test_probs.csv", index=False)


    joblib.dump(calibrator, MODELS_DIR / "xgb_calibrator.joblib")
    print("  Saved: outputs/models/xgb_model.joblib (seed 42)")
    print("  Saved: outputs/models/xgb_oof_predictions.csv")
    print("  Saved: outputs/models/xgb_test_probs.csv")


if __name__ == "__main__":
    main()
