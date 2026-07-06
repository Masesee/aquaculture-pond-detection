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

from contracts.schema import TARGET_COL, WINDOW_METADATA_COLS
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


def correct_prior(probs: np.ndarray, train_prior: float, test_prior: float, allow_shift: bool = False) -> np.ndarray:
    """
    Adjusts probabilities for a class prior shift.
    Formula: p_new = (p * (pi_test / pi_train)) / (p * (pi_test / pi_train) + (1-p) * ((1-pi_test) / (1-pi_train)))
    """
    if not np.isclose(test_prior, train_prior, atol=1e-4):
        if not allow_shift:
            raise ValueError(
                f"Prior shift correction (test_prior={test_prior:.4f} != train_prior={train_prior:.4f}) "
                "is forbidden by default to prevent threshold tuning under Zindi rules. "
                "Set allow_shift=True to bypass if you have an externally sourced population estimate."
            )

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
    test_prior = None
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

    # ── Optional: Exclude window metadata features ────────────────────────────
    exclude_metadata = "--exclude-metadata" in sys.argv
    metadata_cols = WINDOW_METADATA_COLS
    if exclude_metadata:
        feature_cols = [c for c in feature_cols if c not in metadata_cols]
        print(f"  Excluded window metadata features. Remaining: {len(feature_cols)}")
    else:
        for col in metadata_cols:
            if col not in feature_cols and col in train_df.columns:
                feature_cols.append(col)
        print(f"  Included window metadata features. Total features: {len(feature_cols)}")


    # ── Optional: Quantile Transformation ──────────────────────────────────────
    use_quantile = "--no-quantile" not in sys.argv
    if use_quantile:
        from sklearn.preprocessing import QuantileTransformer
        print("  Applying Quantile Transformation separately to train and test sets...")
        qt_train = QuantileTransformer(n_quantiles=1000, random_state=42, output_distribution="normal")
        qt_test = QuantileTransformer(n_quantiles=1000, random_state=42, output_distribution="normal")
        
        train_features_trans = pd.DataFrame(
            qt_train.fit_transform(train_df[feature_cols]),
            columns=feature_cols,
            index=train_df.index
        )
        test_features_trans = pd.DataFrame(
            qt_test.fit_transform(test_df[feature_cols]),
            columns=feature_cols,
            index=test_df.index
        )
        
        for col in feature_cols:
            train_df[col] = train_features_trans[col]
            test_df[col] = test_features_trans[col]
            
        print("  Quantile Transformation complete.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]
    
    train_prior = y_train.mean()
    if test_prior is None:
        test_prior = train_prior
    else:
        if not np.isclose(test_prior, train_prior, atol=1e-4):
            assert "--allow-prior-shift" in sys.argv, (
                f"Prior shift correction (test_prior={test_prior:.4f} != train_prior={train_prior:.4f}) "
                "is forbidden by default to prevent threshold tuning under Zindi rules. "
                "Pass --allow-prior-shift to bypass if you have an externally sourced population estimate."
            )

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Train positive rate (augmented): {train_prior:.4f}")
    print(f"  Adjusting for test positive prior: {test_prior:.4f}")

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

    # ── CV splits (Group-Aware Stratified CV on single-window validation subset) ──
    from pipelines.training.cv_strategy import get_single_window_indices
    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)

    print(f"\n=== {N_SPLITS}-fold stratified group CV (on 1-window-per-sample validation subset) ===")
    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    split_summary = describe_splits(train_df_single, splits)
    print(split_summary.to_string(index=False))

    # ── Optional pseudo-labeling ───────────────────────────────────────────
    use_pseudo = "--pseudo" in sys.argv

    if use_pseudo:
        sub_path = SUBMISSIONS_DIR / "blend_submission_best.csv"
        if not sub_path.exists():
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
        print(f"  Training size: {len(train_df)} -> {len(train_augmented)} (+{n_pseudo} pseudo-labeled)")
    else:
        train_augmented = train_df

    # ── OOF loop ──────────────────────────────────────────────────────────────
    oof_probs   = np.zeros(len(train_df_single), dtype=float)
    fold_scores = []

    for fold, (train_pos, val_pos) in enumerate(splits):
        # val_pos refers to index in train_df_single. Map back to train_df:
        val_idx_in_train_df = single_win_indices[val_pos]
        val_base_ids = set(train_df_single.iloc[val_pos]["ID"].apply(lambda x: x.split("_w")[0]))

        X_val = train_df.iloc[val_idx_in_train_df][feature_cols]
        y_val = train_df.iloc[val_idx_in_train_df][TARGET_COL].values

        if use_pseudo:
            # Train mask: rows in train_df not in validation base IDs
            train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
            pseudo_mask = np.ones(len(pseudo_df), dtype=bool)
            full_train_mask = np.concatenate([train_mask, pseudo_mask])
            X_tr = train_augmented.loc[full_train_mask, feature_cols]
            y_tr = train_augmented.loc[full_train_mask, TARGET_COL].values
        else:
            train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
            X_tr = train_df.loc[train_mask, feature_cols]
            y_tr = train_df.loc[train_mask, TARGET_COL].values

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
    y_train_single = train_df_single[TARGET_COL].values
    oof_f1    = f1_score(y_train_single, oof_preds)
    oof_auc   = roc_auc_score(y_train_single, oof_probs)
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
    calibrator    = fit_calibrator(oof_probs, y_train_single)
    cal_probs_oof = apply_calibrator(calibrator, oof_probs)

    cal_preds_oof = (cal_probs_oof >= 0.5).astype(int)
    cal_f1        = f1_score(y_train_single, cal_preds_oof)
    cal_auc       = roc_auc_score(y_train_single, cal_probs_oof)
    cal_score     = combined_score(cal_f1, cal_auc)
    print(f"  Calibrated OOF — F1={cal_f1:.4f} | AUC={cal_auc:.4f} | Score={cal_score:.4f}")

    cal_summary = calibration_summary(oof_probs, cal_probs_oof, y_train_single)
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

    # Fit final models on multiple seeds and average predictions
    seeds = [42, 100, 2026]
    if "--seeds" in sys.argv:
        try:
            idx = sys.argv.index("--seeds") + 1
            seeds = [int(x) for x in sys.argv[idx].split(",")]
        except (ValueError, IndexError):
            pass
    raw_test_probs_list = []
    
    print(f"\n=== Training final model on full training data (Seed Averaging across {seeds}) ===")
    for seed in seeds:
        seed_params = {**final_params, "random_state": seed}
        model = lgb.LGBMClassifier(**{
            k: v for k, v in seed_params.items()
            if k not in ["metric"]
        })
        model.fit(X_train, y_train)
        raw_test_probs_list.append(model.predict_proba(X_test)[:, 1])
        if seed == 42:
            joblib.dump(model, MODELS_DIR / "lgbm_model.joblib")
            
    raw_test_probs = np.mean(raw_test_probs_list, axis=0)
    cal_test_probs  = apply_calibrator(calibrator, raw_test_probs)

    # Save LightGBM test probabilities to CSV
    lgbm_test_df = pd.DataFrame({
        "ID": test_df["ID"],
        "lgbm_prob_raw": raw_test_probs,
        "lgbm_prob_cal": cal_test_probs,
    })
    lgbm_test_df.to_csv(MODELS_DIR / "lgbm_test_probs.csv", index=False)
    print("  Saved: outputs/models/lgbm_test_probs.csv")

    # Prior shift correction for final submission
    train_prior = y_train.mean()
    allow_shift = "--allow-prior-shift" in sys.argv
    cal_test_probs_corrected = correct_prior(cal_test_probs, train_prior, test_prior, allow_shift=allow_shift)
    binary_preds    = (cal_test_probs_corrected >= 0.5).astype(int)

    print(f"  Test predicted positive rate (corrected): {binary_preds.mean():.3f}")
    print(f"  Test calibrated/corrected prob range:     [{cal_test_probs_corrected.min():.3f}, {cal_test_probs_corrected.max():.3f}]")

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        "ID":          train_df_single["ID"].values,
        "label":       y_train_single,
        "oof_prob_raw": oof_probs,
        "oof_prob_cal": cal_probs_oof,
        "oof_pred":    cal_preds_oof,
    })
    oof_df.to_csv(MODELS_DIR / "oof_predictions.csv", index=False)

    # ── Save models ───────────────────────────────────────────────────────────
    save_calibrator(calibrator, MODELS_DIR / "calibrator.joblib")
    print("  Saved: outputs/models/lgbm_model.joblib")
    print("  Saved: outputs/models/calibrator.joblib")
    if use_quantile:
        joblib.dump(qt_train, MODELS_DIR / "quantile_transformer_train.joblib")
        joblib.dump(qt_test, MODELS_DIR / "quantile_transformer_test.joblib")
        print("  Saved: outputs/models/quantile_transformer_train.joblib")
        print("  Saved: outputs/models/quantile_transformer_test.joblib")

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