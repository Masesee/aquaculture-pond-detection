"""
Training entrypoint.

Loads feature matrices, runs stratified group k-fold CV, trains final LightGBM
on full training data, calibrates probabilities, applies class prior correction
for expected test set prior shift, and saves model + submission.

Run with:
    python -m pipelines.training.train [--test-prior 0.55]

Outputs:
    outputs/models/lgbm_model.joblib
    outputs/models/calibrator.joblib
    outputs/models/oof_predictions.csv
    outputs/models/cv_summary.csv
    outputs/models/calibration_summary.csv
    outputs/submissions/submission.csv
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL
from pipelines.training.cv_strategy import make_cv_splits, describe_splits
from pipelines.training.calibration import (
    fit_calibrator,
    apply_calibrator,
    calibration_summary,
    save_calibrator,
)

PROCESSED_DIR  = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "outputs" / "models"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── LightGBM hyperparameters ───────────────────────────────────────────────────
LGBM_PARAMS: dict = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "n_estimators":     1000,          # high; early stopping controls actual count
    "learning_rate":    0.02,          # low lr + many rounds = better generalisation
    "num_leaves":       31,            # conservative for small data
    "max_depth":        -1,
    "min_child_samples": 20,           # minimum samples per leaf
    "subsample":        0.8,           # row subsampling per tree
    "subsample_freq":   1,
    "colsample_bytree": 0.8,           # feature subsampling per tree
    "reg_alpha":        0.1,           # L1
    "reg_lambda":       1.0,           # L2
    "class_weight":     None,          # do not balance near-balanced data
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

N_SPLITS     = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50


def combined_score(f1: float, auc: float) -> float:
    """Zindi metric: 0.6 * F1 + 0.4 * AUC."""
    return 0.6 * f1 + 0.4 * auc


def correct_prior(probs: np.ndarray, train_prior: float, test_prior: float) -> np.ndarray:
    """
    Adjusts probabilities for a class prior shift.
    Formula: p_new = (p * (pi_test / pi_train)) / (p * (pi_test / pi_train) + (1-p) * ((1-pi_test) / (1-pi_train)))
    """
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0 - eps)
    ratio_pos = test_prior / train_prior
    ratio_neg = (1.0 - test_prior) / (1.0 - train_prior)
    corrected = (probs * ratio_pos) / (probs * ratio_pos + (1.0 - probs) * ratio_neg)
    return np.clip(corrected, 0.0, 1.0)


def load_pseudo_labels(
    submission_path: Path,
    test_features_path: Path,
    high_conf_threshold: float = 0.95,
    low_conf_threshold: float  = 0.05,
) -> tuple[pd.DataFrame, int]:
    """
    Loads high-confidence test predictions as pseudo-labeled training data.
    """
    sub      = pd.read_csv(submission_path)
    test_df  = pd.read_parquet(test_features_path)

    # Align on ID
    merged = test_df.merge(sub[["ID", "TargetRAUC"]], on="ID", how="left")

    high_mask = merged["TargetRAUC"] > high_conf_threshold
    low_mask  = merged["TargetRAUC"] < low_conf_threshold

    pseudo = merged[high_mask | low_mask].copy()
    pseudo[TARGET_COL] = (pseudo["TargetRAUC"] > high_conf_threshold).astype(int)
    pseudo = pseudo.drop(columns=["TargetRAUC"])

    print(f"  Pseudo-labels: {high_mask.sum()} positive + {low_mask.sum()} negative "
          f"= {len(pseudo)} total")
    print(f"  Pseudo positive rate: {pseudo[TARGET_COL].mean():.3f}")

    return pseudo, len(pseudo)


def main() -> None:
    # Parse CLI Arguments
    test_prior = 0.55  # Default tuned for expected prior shift
    if "--test-prior" in sys.argv:
        try:
            idx = sys.argv.index("--test-prior") + 1
            test_prior = float(sys.argv[idx])
        except (ValueError, IndexError):
            pass

    print("=== Loading feature matrices ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    if invariant_path.exists():
        with open(invariant_path) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(feature_cols)} robust features from invariant_features.txt")
    else:
        feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]
        print(f"  Using all {len(feature_cols)} features (no invariant_features.txt found)")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Train positive rate (augmented): {y_train.mean():.3f}")
    print(f"  Adjusting for test positive prior: {test_prior:.3f}")

    # ── Load tuned params if available ────────────────────────────────────────
    best_params_path = MODELS_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            tuned = json.load(f)
        active_params = {
            **LGBM_PARAMS,
            **tuned,
            "objective":     "binary",
            "class_weight":  None,
            "random_state":  42,
            "n_jobs":        -1,
            "verbose":       -1,
        }
        print(f"  Loaded tuned params from {best_params_path.name}")
    else:
        active_params = LGBM_PARAMS
        print("  Using default params (no tuning found)")

    # ── CV splits (Group-Aware Stratified CV) ─────────────────────────────────
    print(f"\n=== {N_SPLITS}-fold stratified group CV (grouped by original ID) ===")
    splits = make_cv_splits(train_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    split_summary = describe_splits(train_df, splits)
    print(split_summary.to_string(index=False))

    # ── Optional pseudo-labeling ───────────────────────────────────────────
    use_pseudo = "--pseudo" in sys.argv

    if use_pseudo:
        sub_path = SUBMISSIONS_DIR / "submission.csv"
        if not sub_path.exists():
            print(f"\nERROR: Pseudo-labeling requires a source submission at: {sub_path}")
            print("Please run the pipeline once WITHOUT the --pseudo flag first.")
            return

        print("\n=== Loading pseudo-labels ===")
        pseudo_df, n_pseudo = load_pseudo_labels(
            submission_path    = sub_path,
            test_features_path = PROCESSED_DIR   / "test_features.parquet",
        )
        train_augmented = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"  Training size: {len(train_df)} → {len(train_augmented)} (+{n_pseudo} pseudo-labeled)")

        X_train_full = train_augmented[feature_cols]
        y_train_full = train_augmented[TARGET_COL].values

        # pseudo rows are always in train, never in val
        splits_for_pseudo = [
            (
                np.concatenate([tr, np.arange(len(train_df), len(train_augmented))]),
                val
            )
            for tr, val in splits
        ]
    else:
        X_train_full     = X_train
        y_train_full     = y_train
        splits_for_pseudo = splits

    # ── OOF loop ──────────────────────────────────────────────────────────────
    oof_probs   = np.zeros(len(train_df), dtype=float)
    fold_scores = []

    for fold, (train_pos, val_pos) in enumerate(splits_for_pseudo):
        X_tr, y_tr = X_train_full.iloc[train_pos], y_train_full[train_pos]
        X_val, y_val = X_train_full.iloc[val_pos], y_train_full[val_pos]

        # CV models use early stopping
        cv_params = {k: v for k, v in active_params.items() if k != "n_estimators"}
        model = lgb.LGBMClassifier(**{**cv_params, "n_estimators": 1000})

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        fold_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_pos] = fold_probs

        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_f1    = f1_score(y_val, fold_preds)
        fold_auc   = roc_auc_score(y_val, fold_probs)
        fold_score = combined_score(fold_f1, fold_auc)

        fold_scores.append({
            "fold":       fold,
            "best_iter":  model.best_iteration_,
            "f1":         round(fold_f1,    4),
            "auc":        round(fold_auc,   4),
            "score":      round(fold_score, 4),
        })
        print(
            f"  Fold {fold}: iter={model.best_iteration_:4d} | "
            f"F1={fold_f1:.4f} | AUC={fold_auc:.4f} | Score={fold_score:.4f}"
        )

    # ── OOF aggregate metrics (without prior shift correction, evaluated on OOF distribution)
    oof_preds = (oof_probs >= 0.5).astype(int)
    oof_f1    = f1_score(y_train, oof_preds)
    oof_auc   = roc_auc_score(y_train, oof_probs)
    oof_score = combined_score(oof_f1, oof_auc)

    print(f"\n  OOF aggregate (pre-cal) — F1={oof_f1:.4f} | AUC={oof_auc:.4f} | Score={oof_score:.4f}")

    cv_df = pd.DataFrame(fold_scores)
    cv_df.loc[len(cv_df)] = {
        "fold": "OOF", "best_iter": None,
        "f1": round(oof_f1, 4), "auc": round(oof_auc, 4), "score": round(oof_score, 4),
    }
    cv_df.to_csv(MODELS_DIR / "cv_summary.csv", index=False)

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\n=== Fitting calibrator on OOF predictions ===")
    calibrator    = fit_calibrator(oof_probs, y_train)
    cal_probs_oof = apply_calibrator(calibrator, oof_probs)

    cal_preds_oof = (cal_probs_oof >= 0.5).astype(int)
    cal_f1        = f1_score(y_train, cal_preds_oof)
    cal_auc       = roc_auc_score(y_train, cal_probs_oof)
    cal_score     = combined_score(cal_f1, cal_auc)
    print(f"  Calibrated OOF — F1={cal_f1:.4f} | AUC={cal_auc:.4f} | Score={cal_score:.4f}")

    cal_summary = calibration_summary(oof_probs, cal_probs_oof, y_train)
    cal_summary.to_csv(MODELS_DIR / "calibration_summary.csv", index=False)
    print("  Calibration summary saved.")

    # ── Final model: retrain on full training data ────────────────────────────
    print("\n=== Training final model on full training data ===")
    best_iters = [s["best_iter"] for s in fold_scores if s["best_iter"] is not None]

    if best_params_path.exists():
        with open(best_params_path) as f:
            _bp = json.load(f)
        if "n_estimators" in _bp:
            final_n_estimators = _bp["n_estimators"]
            print(f"  Using Optuna n_estimators={final_n_estimators} for final fit")
        else:
            final_n_estimators = int(round(np.mean(best_iters) * 1.05)) if best_iters else 100
            print(f"  CV best_iter mean={np.mean(best_iters):.0f} → final n_estimators={final_n_estimators}")
    else:
        final_n_estimators = int(round(np.mean(best_iters) * 1.05)) if best_iters else 100
        print(f"  CV best_iter mean={np.mean(best_iters):.0f} → final n_estimators={final_n_estimators}")

    final_params = {**active_params, "n_estimators": final_n_estimators}
    final_model = lgb.LGBMClassifier(**{
        k: v for k, v in final_params.items()
        if k not in ["metric"]
    })
    final_model.fit(X_train, y_train)

    # ── Generate test predictions ─────────────────────────────────────────────
    print("\n=== Generating test predictions ===")
    raw_test_probs  = final_model.predict_proba(X_test)[:, 1]
    cal_test_probs  = apply_calibrator(calibrator, raw_test_probs)

    # Prior shift correction for final submission
    train_prior = y_train.mean()
    cal_test_probs_corrected = correct_prior(cal_test_probs, train_prior, test_prior)
    binary_preds    = (cal_test_probs_corrected >= 0.5).astype(int)

    print(f"  Test predicted positive rate (corrected): {binary_preds.mean():.3f}")
    print(f"  Test calibrated/corrected prob range:     [{cal_test_probs_corrected.min():.3f}, {cal_test_probs_corrected.max():.3f}]")

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        "ID":          train_df["ID"].values,
        "label":       y_train,
        "oof_prob_raw": oof_probs,
        "oof_prob_cal": cal_probs_oof,
        "oof_pred":    oof_preds,
    })
    oof_df.to_csv(MODELS_DIR / "oof_predictions.csv", index=False)

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(final_model, MODELS_DIR / "lgbm_model.joblib")
    save_calibrator(calibrator, MODELS_DIR / "calibrator.joblib")
    print("  Saved: outputs/models/lgbm_model.joblib")
    print("  Saved: outputs/models/calibrator.joblib")

    # ── Build submission ──────────────────────────────────────────────────────
    print("\n=== Building submission file ===")
    submission = pd.DataFrame({
        "ID":          test_df["ID"].values,
        "TargetF1":    binary_preds,
        "TargetRAUC":  cal_test_probs_corrected.round(6),
    })
    submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)
    print("  Saved: outputs/submissions/submission.csv")
    print(f"  Submission shape: {submission.shape}")
    print(f"  Predicted ponds: {binary_preds.sum()} / {len(binary_preds)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n=== Training complete ===")
    print(f"  OOF Score (pre-cal):  {oof_score:.4f}")
    print(f"  OOF Score (post-cal): {cal_score:.4f}")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    main()