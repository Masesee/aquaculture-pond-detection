"""
Stacking ensemble entrypoint.

Architecture:
  Layer 0 (base learners): LightGBM, RandomForest, LogisticRegression
    - Each trained with stratified k-fold CV
    - OOF predictions collected for each learner
  Layer 1 (meta-learner): LogisticRegression
    - Trained on stacked OOF predictions from all base learners
    - Learns when to trust each base learner

Why this works:
  Each base learner makes different errors. LightGBM overfits slightly
  to tree-splittable boundaries. RF uses a different split heuristic and
  averaging. LogReg is constrained to linear water index signal. The
  meta-learner discovers that when LogReg disagrees with LightGBM, RF
  is usually the tiebreaker.

Run with:
    python -m pipelines.training.ensemble

Outputs:
    outputs/models/ensemble_oof.csv
    outputs/models/ensemble_cv_summary.csv
    outputs/models/ensemble_meta_learner.joblib
    outputs/models/ensemble_base_lgbm.joblib
    outputs/models/ensemble_base_rf.joblib
    outputs/models/ensemble_base_logreg.joblib
    outputs/submissions/submission_ensemble.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL
from pipelines.training.cv_strategy import make_cv_splits, describe_splits
from pipelines.training.calibration import fit_calibrator, apply_calibrator
from pipelines.training.base_learners import (
    make_lgbm, make_rf, make_logreg,
    LOGREG_FEATURES, EARLY_STOPPING_ROUNDS,
)
from pipelines.evaluation.metrics import combined_score, evaluate

PROCESSED_DIR   = ROOT / "data"    / "processed"
MODELS_DIR      = ROOT / "outputs" / "models"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS     = 5
RANDOM_STATE = 42


# ── Base learner OOF loop ─────────────────────────────────────────────────────

def _fit_lgbm_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list,
) -> tuple[np.ndarray, list[lgb.LGBMClassifier]]:
    """
    Runs LightGBM through k-fold CV with early stopping.
    Returns OOF probabilities and list of fold models.
    """
    oof     = np.zeros(len(y))
    models  = []
    iters   = []

    for fold, (tr, val) in enumerate(splits):
        model = make_lgbm()
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[val], y[val])],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        models.append(model)
        iters.append(model.best_iteration_)
        _print_fold("LGBM", fold, y[val], oof[val])

    print(f"  LGBM mean best_iter: {np.mean(iters):.0f}")
    return oof, models


def _fit_rf_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list,
) -> tuple[np.ndarray, list]:
    """
    Runs RandomForest through k-fold CV.
    RF has no early stopping — fits fully each fold.
    """
    oof    = np.zeros(len(y))
    models = []

    for fold, (tr, val) in enumerate(splits):
        model = make_rf()
        model.fit(X.iloc[tr], y[tr])
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        models.append(model)
        _print_fold("RF  ", fold, y[val], oof[val])

    return oof, models


def _fit_logreg_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list,
    feature_cols: list[str],
) -> tuple[np.ndarray, list]:
    """
    Runs LogisticRegression through k-fold CV on LOGREG_FEATURES subset.
    Scaler is fit inside each fold to prevent leakage.
    """
    # Only keep features that exist in X
    logreg_cols = [f for f in LOGREG_FEATURES if f in feature_cols]
    missing     = set(LOGREG_FEATURES) - set(logreg_cols)
    if missing:
        print(f"  LogReg: {len(missing)} requested features not in matrix, skipping them")
    print(f"  LogReg using {len(logreg_cols)} features")

    X_lr   = X[logreg_cols]
    oof    = np.zeros(len(y))
    models = []

    for fold, (tr, val) in enumerate(splits):
        model = make_logreg()
        model.fit(X_lr.iloc[tr], y[tr])
        oof[val] = model.predict_proba(X_lr.iloc[val])[:, 1]
        models.append(model)
        _print_fold("LR  ", fold, y[val], oof[val])

    return oof, models, logreg_cols


def _print_fold(name: str, fold: int, y_val: np.ndarray, probs: np.ndarray) -> None:
    preds = (probs >= 0.5).astype(int)
    f1    = f1_score(y_val, preds)
    auc   = roc_auc_score(y_val, probs)
    score = combined_score(f1, auc)
    print(f"    {name} Fold {fold}: F1={f1:.4f} | AUC={auc:.4f} | Score={score:.4f}")


# ── Final base model refit ────────────────────────────────────────────────────

def _refit_lgbm_final(
    X: pd.DataFrame,
    y: np.ndarray,
    fold_models: list[lgb.LGBMClassifier],
) -> lgb.LGBMClassifier:
    """Refits LightGBM on full data using mean best_iter × 1.05."""
    mean_iter     = np.mean([m.best_iteration_ for m in fold_models])
    n_estimators  = int(round(mean_iter * 1.05))
    print(f"  LGBM final n_estimators: {n_estimators}")
    model = make_lgbm(n_estimators=n_estimators)
    model.fit(X, y)
    return model


def _refit_rf_final(X: pd.DataFrame, y: np.ndarray) -> object:
    model = make_rf()
    model.fit(X, y)
    return model


def _refit_logreg_final(
    X: pd.DataFrame,
    y: np.ndarray,
    logreg_cols: list[str],
) -> object:
    model = make_logreg()
    model.fit(X[logreg_cols], y)
    return model


# ── Meta-learner ──────────────────────────────────────────────────────────────

def _fit_meta_learner(
    oof_lgbm: np.ndarray,
    oof_rf: np.ndarray,
    oof_lr: np.ndarray,
    y: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Fits the meta-learner on stacked OOF predictions.

    Input features to meta-learner (per sample):
      [lgbm_prob, rf_prob, lr_prob, lgbm_rf_diff, lgbm_lr_diff, rf_lr_diff]

    The difference features capture disagreement between learners —
    high disagreement cases are exactly the hard boundary cases the
    meta-learner needs to resolve correctly.
    """
    meta_X = _build_meta_features(oof_lgbm, oof_rf, oof_lr)

    scaler = StandardScaler()
    meta_X_scaled = scaler.fit_transform(meta_X)

    meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_clf.fit(meta_X_scaled, y)

    # Report meta-learner OOF score
    meta_probs = meta_clf.predict_proba(meta_X_scaled)[:, 1]
    meta_preds = (meta_probs >= 0.5).astype(int)
    f1    = f1_score(y, meta_preds)
    auc   = roc_auc_score(y, meta_probs)
    score = combined_score(f1, auc)
    print(f"  Meta-learner (train): F1={f1:.4f} | AUC={auc:.4f} | Score={score:.4f}")

    # Coefficients — shows which base learner the meta trusts most
    print(f"  Meta coefs [lgbm, rf, lr, lgbm-rf, lgbm-lr, rf-lr]: "
          f"{meta_clf.coef_[0].round(3)}")

    return meta_clf, scaler


def _build_meta_features(
    lgbm_p: np.ndarray,
    rf_p: np.ndarray,
    lr_p: np.ndarray,
) -> np.ndarray:
    """
    Stacks base learner probabilities + pairwise disagreement features.
    Shape: (n, 6)
    """
    return np.column_stack([
        lgbm_p,
        rf_p,
        lr_p,
        lgbm_p - rf_p,    # LGBM vs RF disagreement
        lgbm_p - lr_p,    # LGBM vs LogReg disagreement
        rf_p   - lr_p,    # RF vs LogReg disagreement
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    print("=== Loading feature matrices ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]
    X      = train_df[feature_cols]
    y      = train_df[TARGET_COL].values
    X_test = test_df[feature_cols]

    print(f"  X_train: {X.shape} | X_test: {X_test.shape}")
    print(f"  Positive rate: {y.mean():.3f}")

    # ── CV splits — same strategy as single model ──────────────────────────
    print(f"\n=== Generating {N_SPLITS}-fold splits ===")
    splits = make_cv_splits(train_df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    print(describe_splits(train_df, splits).to_string(index=False))

    # ── Layer 0: base learner OOF ──────────────────────────────────────────
    print("\n=== Base learner OOF — LightGBM ===")
    oof_lgbm, lgbm_fold_models = _fit_lgbm_oof(X, y, splits)

    print("\n=== Base learner OOF — Random Forest ===")
    oof_rf, rf_fold_models = _fit_rf_oof(X, y, splits)

    print("\n=== Base learner OOF — Logistic Regression ===")
    oof_lr, lr_fold_models, logreg_cols = _fit_logreg_oof(X, y, splits, feature_cols)

    # ── Per-learner aggregate OOF scores ──────────────────────────────────
    print("\n=== Base learner OOF aggregate ===")
    for name, oof in [("LGBM", oof_lgbm), ("RF  ", oof_rf), ("LR  ", oof_lr)]:
        result = evaluate(y, oof)
        print(f"  {name}: F1={result['f1']:.4f} | AUC={result['auc']:.4f} | "
              f"Score={result['score']:.4f}")

    # ── Layer 1: meta-learner ──────────────────────────────────────────────
    print("\n=== Fitting meta-learner ===")
    meta_clf, meta_scaler = _fit_meta_learner(oof_lgbm, oof_rf, oof_lr, y)

    # Ensemble OOF score via meta-learner
    meta_X_oof    = _build_meta_features(oof_lgbm, oof_rf, oof_lr)
    meta_X_scaled = meta_scaler.transform(meta_X_oof)
    oof_ensemble  = meta_clf.predict_proba(meta_X_scaled)[:, 1]
    oof_result    = evaluate(y, oof_ensemble)
    print(f"\n  Ensemble OOF (pre-cal): "
          f"F1={oof_result['f1']:.4f} | AUC={oof_result['auc']:.4f} | "
          f"Score={oof_result['score']:.4f}")

    # ── Calibration on ensemble OOF ────────────────────────────────────────
    print("\n=== Calibrating ensemble OOF probabilities ===")
    calibrator     = fit_calibrator(oof_ensemble, y)
    oof_cal        = apply_calibrator(calibrator, oof_ensemble)
    cal_result     = evaluate(y, oof_cal)
    print(f"  Ensemble OOF (post-cal): "
          f"F1={cal_result['f1']:.4f} | AUC={cal_result['auc']:.4f} | "
          f"Score={cal_result['score']:.4f}")

    # ── Refit base learners on full training data ──────────────────────────
    print("\n=== Refitting base learners on full training data ===")
    final_lgbm   = _refit_lgbm_final(X, y, lgbm_fold_models)
    final_rf     = _refit_rf_final(X, y)
    final_logreg = _refit_logreg_final(X, y, logreg_cols)
    print("  All base learners refitted.")

    # ── Generate test predictions ──────────────────────────────────────────
    print("\n=== Generating test predictions ===")
    test_lgbm_p = final_lgbm.predict_proba(X_test)[:, 1]
    test_rf_p   = final_rf.predict_proba(X_test)[:, 1]
    test_lr_p   = final_logreg.predict_proba(X_test[logreg_cols])[:, 1]

    meta_X_test   = _build_meta_features(test_lgbm_p, test_rf_p, test_lr_p)
    meta_X_t_scl  = meta_scaler.transform(meta_X_test)
    raw_test_probs = meta_clf.predict_proba(meta_X_t_scl)[:, 1]
    cal_test_probs = apply_calibrator(calibrator, raw_test_probs)
    binary_preds   = (cal_test_probs >= 0.5).astype(int)

    print(f"  Test predicted positive rate: {binary_preds.mean():.3f}")
    print(f"  Test calibrated prob range:   "
          f"[{cal_test_probs.min():.3f}, {cal_test_probs.max():.3f}]")
    print(f"  Predicted ponds: {binary_preds.sum()} / {len(binary_preds)}")

    # ── Save OOF predictions ───────────────────────────────────────────────
    oof_df = pd.DataFrame({
        "ID":           train_df["ID"].values,
        "label":        y,
        "oof_lgbm":     oof_lgbm,
        "oof_rf":       oof_rf,
        "oof_lr":       oof_lr,
        "oof_ensemble": oof_ensemble,
        "oof_cal":      oof_cal,
    })
    oof_df.to_csv(MODELS_DIR / "ensemble_oof.csv", index=False)

    # ── Save CV summary ────────────────────────────────────────────────────
    cv_rows = []
    for name, oof in [("LGBM", oof_lgbm), ("RF", oof_rf), ("LR", oof_lr),
                      ("Ensemble", oof_ensemble), ("Ensemble_cal", oof_cal)]:
        r = evaluate(y, oof)
        cv_rows.append({"learner": name, **r})
    pd.DataFrame(cv_rows).to_csv(MODELS_DIR / "ensemble_cv_summary.csv", index=False)

    # ── Save models ────────────────────────────────────────────────────────
    joblib.dump(final_lgbm,   MODELS_DIR / "ensemble_base_lgbm.joblib")
    joblib.dump(final_rf,     MODELS_DIR / "ensemble_base_rf.joblib")
    joblib.dump(final_logreg, MODELS_DIR / "ensemble_base_logreg.joblib")
    joblib.dump(meta_clf,     MODELS_DIR / "ensemble_meta_learner.joblib")
    joblib.dump(meta_scaler,  MODELS_DIR / "ensemble_meta_scaler.joblib")
    joblib.dump(calibrator,   MODELS_DIR / "ensemble_calibrator.joblib")
    joblib.dump(logreg_cols,  MODELS_DIR / "ensemble_logreg_cols.joblib")
    print("  All models saved.")

    # ── Build submission ───────────────────────────────────────────────────
    submission = pd.DataFrame({
        "ID":         test_df["ID"].values,
        "TargetF1":   binary_preds,
        "TargetRAUC": cal_test_probs.round(6),
    })
    submission.to_csv(SUBMISSIONS_DIR / "submission_ensemble.csv", index=False)
    print("\n  Saved: outputs/submissions/submission_ensemble.csv")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n=== Ensemble complete ===")
    print(f"  Single LGBM OOF:    {evaluate(y, oof_lgbm)['score']:.4f}")
    print(f"  Ensemble OOF:       {oof_result['score']:.4f}")
    print(f"  Ensemble OOF (cal): {cal_result['score']:.4f}")
    delta = cal_result['score'] - evaluate(y, oof_lgbm)['score']
    print(f"  Delta vs LGBM:      {delta:+.4f}")


if __name__ == "__main__":
    main()