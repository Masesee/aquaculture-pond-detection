"""
Training entrypoint.

Loads feature matrices, runs stratified k-fold CV, trains final LightGBM
on full training data, calibrates probabilities, saves model + submission.

Run with:
    python -m pipelines.training.train

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
# Starting point: conservative regularization for 963 samples / 169 features.
# These will be tuned in the next pipeline step (Optuna sweep).
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
    "class_weight":     "balanced",    # handles any residual imbalance
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


def load_pseudo_labels(
    submission_path: Path,
    test_features_path: Path,
    high_conf_threshold: float = 0.95,
    low_conf_threshold: float  = 0.05,
) -> tuple[pd.DataFrame, int]:
    """
    Loads high-confidence test predictions as pseudo-labeled training data.

    Parameters
    ----------
    submission_path      : path to current best submission CSV
    test_features_path   : path to test_features.parquet
    high_conf_threshold  : predictions above this → pseudo-label 1
    low_conf_threshold   : predictions below this → pseudo-label 0

    Returns
    -------
    pseudo_df  : feature DataFrame with TARGET_COL column appended
    n_pseudo   : number of pseudo-labeled samples added
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


def run_pseudo_iterations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    active_params: dict,
    n_iterations: int,
    high_threshold: float = 0.95,
    low_threshold: float  = 0.05,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Runs iterative pseudo-labeling for n_iterations rounds.

    Each round:
      1. Trains on labeled + current pseudo-labeled data
      2. Predicts on full test set
      3. Updates pseudo-label pool with newly confident predictions
      4. Stops early if pseudo-label set does not change

    Returns
    -------
    final_train_df : augmented training DataFrame for final model fit
    final_test_probs : raw test probabilities from last iteration
    """
    from pipelines.training.calibration import fit_calibrator, apply_calibrator

    current_train = train_df.copy()
    prev_pseudo_ids: set = set()

    for iteration in range(1, n_iterations + 1):
        print(f"\n  === Pseudo iteration {iteration}/{n_iterations} ===")

        X_iter = current_train[feature_cols]
        y_iter = current_train[TARGET_COL].values

        # CV splits on original indices only
        splits = make_cv_splits(
            train_df,   # always original — val indices must stay clean
            n_splits=n_splits,
            random_state=random_state,
        )
        # Extend train portion of each fold with pseudo rows
        n_orig = len(train_df)
        n_aug  = len(current_train)
        pseudo_indices = np.arange(n_orig, n_aug)

        splits_aug = [
            (np.concatenate([tr, pseudo_indices]), val)
            for tr, val in splits
        ]

        # OOF loop
        oof_probs = np.zeros(len(train_df))
        best_iters = []

        for fold, (tr, val) in enumerate(splits_aug):
            model = lgb.LGBMClassifier(**{
                **active_params, "n_estimators": 1000
            })
            model.fit(
                X_iter.iloc[tr], y_iter[tr],
                eval_set=[(X_iter.iloc[val], y_iter[val])],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            oof_probs[val] = model.predict_proba(
                X_iter.iloc[val]
            )[:, 1]
            best_iters.append(model.best_iteration_)

        oof_preds = (oof_probs >= 0.5).astype(int)
        f1  = f1_score(train_df[TARGET_COL].values, oof_preds)
        auc = roc_auc_score(train_df[TARGET_COL].values, oof_probs)
        print(f"    OOF F1={f1:.4f} | AUC={auc:.4f} | "
              f"Score={combined_score(f1, auc):.4f}")

        # Refit on full augmented set
        n_est = int(round(np.mean(best_iters) * 1.05))
        final = lgb.LGBMClassifier(**{**active_params, "n_estimators": n_est})
        final.fit(X_iter, y_iter)

        # Predict on test
        raw_test = final.predict_proba(test_df[feature_cols])[:, 1]

        # Calibrate
        calibrator = fit_calibrator(oof_probs, train_df[TARGET_COL].values)
        cal_test   = apply_calibrator(calibrator, raw_test)

        # Build new pseudo-label pool
        new_pseudo_ids_pos = set(
            test_df.loc[cal_test > high_threshold, "ID"].tolist()
        )
        new_pseudo_ids_neg = set(
            test_df.loc[cal_test < low_threshold,  "ID"].tolist()
        )
        new_pseudo_ids = new_pseudo_ids_pos | new_pseudo_ids_neg

        added   = len(new_pseudo_ids - prev_pseudo_ids)
        removed = len(prev_pseudo_ids - new_pseudo_ids)
        print(f"    Pseudo pool: {len(new_pseudo_ids)} "
              f"(+{added} added, -{removed} removed vs prev iteration)")

        # Early stop if pool stabilized
        if new_pseudo_ids == prev_pseudo_ids and iteration > 1:
            print(f"    Pseudo pool stable — stopping early at iteration {iteration}")
            break

        prev_pseudo_ids = new_pseudo_ids

        # Rebuild augmented training set
        pseudo_mask = (cal_test > high_threshold) | (cal_test < low_threshold)
        pseudo_rows = test_df[pseudo_mask].copy()
        pseudo_rows[TARGET_COL] = (cal_test[pseudo_mask] > high_threshold).astype(int)
        pseudo_rows = pseudo_rows.drop(
            columns=[c for c in pseudo_rows.columns if c not in
                     [col for col in feature_cols] + ["ID", TARGET_COL]]
        )

        current_train = pd.concat(
            [train_df, pseudo_rows], ignore_index=True
        )
        print(f"    Training size: {len(current_train)} "
              f"({len(pseudo_rows)} pseudo-labeled)")

    return current_train, cal_test


def analyze_confidence_distribution(
    submission_path: Path,
) -> None:
    """Prints confidence distribution to help choose pseudo-label threshold."""
    sub   = pd.read_csv(submission_path)
    probs = sub["TargetRAUC"].values
    print("  Prob distribution:")
    for threshold in [0.99, 0.95, 0.90, 0.85, 0.80]:
        n_high = (probs > threshold).sum()
        n_low  = (probs < 1 - threshold).sum()
        print(f"    >{threshold:.2f} or <{1-threshold:.2f}: "
              f"{n_high} positive + {n_low} negative = {n_high+n_low} total "
              f"({100*(n_high+n_low)/len(probs):.1f}% of test)")


def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    print("=== Loading feature matrices ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Positive rate: {y_train.mean():.3f}")

    # ── Load tuned params if available ────────────────────────────────────────
    best_params_path = MODELS_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            tuned = json.load(f)
        # Merge: tuned params override defaults, infrastructure params preserved
        active_params = {
            **LGBM_PARAMS,
            **tuned,
            "objective":     "binary",
            # "class_weight":  "balanced",
            "class_weight":  None,
            "random_state":  42,
            "n_jobs":        -1,
            "verbose":       -1,
        }
        print(f"  Loaded tuned params from {best_params_path.name}")
    else:
        active_params = LGBM_PARAMS
        print("  Using default params (no tuning found)")

    # ── CV splits ─────────────────────────────────────────────────────────────
    print(f"\n=== {N_SPLITS}-fold stratified CV (label × region) ===")
    splits = make_cv_splits(train_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    split_summary = describe_splits(train_df, splits)
    print(split_summary.to_string(index=False))

    # ── Optional iterative pseudo-labeling ────────────────────────────────────
    import sys as _sys
    use_pseudo_iter = "--pseudo-iter" in _sys.argv
    if use_pseudo_iter:
        n_iter_idx = _sys.argv.index("--pseudo-iter") + 1
        n_iter     = int(_sys.argv[n_iter_idx]) if n_iter_idx < len(_sys.argv) else 3

        print(f"\n=== Iterative pseudo-labeling ({n_iter} rounds) ===")
        train_augmented, cal_test_probs_iter = run_pseudo_iterations(
            train_df      = train_df,
            test_df       = test_df,
            feature_cols  = feature_cols,
            active_params = active_params,
            n_iterations  = n_iter,
        )
        # Use augmented data for final model — skip the normal train loop
        # Write submission directly from last iteration's cal_test_probs_iter
        binary_preds = (cal_test_probs_iter >= 0.5).astype(int)
        submission = pd.DataFrame({
            "ID":         test_df["ID"].values,
            "TargetF1":   binary_preds,
            "TargetRAUC": cal_test_probs_iter.round(6),
        })
        submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)
        print(f"\n  Predicted ponds: {binary_preds.sum()} / {len(binary_preds)}")
        print("  Saved: outputs/submissions/submission.csv")
        return   # skip normal training flow

    # ── Optional pseudo-labeling ───────────────────────────────────────────
    use_pseudo = "--pseudo" in _sys.argv

    if use_pseudo:
        sub_path = SUBMISSIONS_DIR / "submission.csv"
        if not sub_path.exists():
            print(f"\nERROR: Pseudo-labeling requires a source submission at: {sub_path}")
            print("Please run the pipeline once WITHOUT the --pseudo flag first.")
            return

        print("\n=== Confidence distribution in current submission ===")
        analyze_confidence_distribution(sub_path)

        print("\n=== Loading pseudo-labels ===")
        pseudo_df, n_pseudo = load_pseudo_labels(
            submission_path    = sub_path,
            test_features_path = PROCESSED_DIR   / "test_features.parquet",
        )
        # Append pseudo-labeled rows to training data
        train_augmented = pd.concat(
            [train_df, pseudo_df], ignore_index=True
        )
        print(f"  Training size: {len(train_df)} → {len(train_augmented)} "
              f"(+{n_pseudo} pseudo-labeled)")

        # Rebuild X_train and y_train from augmented set
        # CV is run on ORIGINAL indices only — pseudo labels go into train folds
        # but never into validation folds. This prevents pseudo-label leakage.
        X_train_full = train_augmented[feature_cols]
        y_train_full = train_augmented[TARGET_COL].values

        # Splits still generated on original train_df only
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

        model = lgb.LGBMClassifier(**active_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),   # silent
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

    # ── OOF aggregate metrics ─────────────────────────────────────────────────
    oof_preds = (oof_probs >= 0.5).astype(int)
    oof_f1    = f1_score(y_train, oof_preds)
    oof_auc   = roc_auc_score(y_train, oof_probs)
    oof_score = combined_score(oof_f1, oof_auc)

    print(f"\n  OOF aggregate — F1={oof_f1:.4f} | AUC={oof_auc:.4f} | Score={oof_score:.4f}")

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
    best_iters = [s["best_iter"] for s in fold_scores]
    final_n_estimators = int(round(np.mean(best_iters) * 1.05))  # 5% more rounds than CV mean
    print(f"  CV best_iter mean={np.mean(best_iters):.0f} → final n_estimators={final_n_estimators}")

    final_params = {**active_params, "n_estimators": final_n_estimators}
    # Remove early stopping keys not valid without eval_set
    final_model = lgb.LGBMClassifier(**{
        k: v for k, v in final_params.items()
        if k not in ["metric"]
    })
    final_model.fit(X_train, y_train)

    # ── Generate test predictions ─────────────────────────────────────────────
    print("\n=== Generating test predictions ===")
    raw_test_probs  = final_model.predict_proba(X_test)[:, 1]
    cal_test_probs  = apply_calibrator(calibrator, raw_test_probs)
    binary_preds    = (cal_test_probs >= 0.5).astype(int)

    print(f"  Test predicted positive rate: {binary_preds.mean():.3f}")
    print(f"  Test calibrated prob range:   [{cal_test_probs.min():.3f}, {cal_test_probs.max():.3f}]")

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
        "TargetRAUC":  cal_test_probs.round(6),
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