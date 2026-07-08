"""
Optuna hyperparameter sweep for XGBoost on masked temporal features.
Optimises the Zindi combined score (0.6*F1 + 0.4*AUC) on OOF predictions
using StratifiedGroupKFold to prevent data leakage.

Run with:
    python -m pipelines.training.tune_xgb
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL, WINDOW_METADATA_COLS
from pipelines.training.cv_strategy import make_cv_splits, get_single_window_indices, get_fold_train_mask
from pipelines.evaluation.metrics import combined_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR  = ROOT / "data"        / "processed"
LOGS_DIR       = ROOT / "experiments" / "logs"
MODELS_DIR     = ROOT / "outputs"     / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS      = 5
RANDOM_STATE  = 42
N_TRIALS      = 30
STUDY_NAME    = "xgb_aquaculture_regularized"
STORAGE       = f"sqlite:///{LOGS_DIR / 'optuna_study_xgb.db'}"


def objective(trial: optuna.Trial, train_df: pd.DataFrame, train_df_single: pd.DataFrame, feature_cols: list[str], single_win_indices: np.ndarray) -> float:
    params = {
        "n_estimators":      trial.suggest_int(  "n_estimators",      150,    450),
        "learning_rate":     trial.suggest_float("learning_rate",      0.02,   0.08, log=True),
        "max_depth":         trial.suggest_int(  "max_depth",          3,      6),
        "min_child_weight":  trial.suggest_int(  "min_child_weight",   2,      15),
        "subsample":         trial.suggest_float("subsample",          0.55,   0.85),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",   0.35,   0.65),
        "reg_alpha":         trial.suggest_float("reg_alpha",          0.01,   10.0,  log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",         0.1,    30.0,  log=True),
        "eval_metric":       "logloss",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }

    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    y_train_single = train_df_single[TARGET_COL].values
    oof_probs = np.zeros(len(y_train_single), dtype=float)
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])

    for train_pos, val_pos in splits:
        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values

        X_val = train_df_single[feature_cols].iloc[val_pos]
        y_val = y_train_single[val_pos]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_probs[val_pos] = model.predict_proba(X_val)[:, 1]

    preds = (oof_probs >= 0.5).astype(int)
    f1    = f1_score(y_train_single, preds)
    auc   = roc_auc_score(y_train_single, oof_probs)
    return combined_score(f1, auc)


def main() -> None:
    print("=== Loading features ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    
    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    with open(invariant_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
        
    metadata_cols = WINDOW_METADATA_COLS
    for col in metadata_cols:
        if col not in feature_cols and col in train_df.columns:
            feature_cols.append(col)
            
    print(f"  Loaded {len(feature_cols)} features (invariant + metadata)")

    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)

    print(f"\n=== Running Optuna for XGBoost ({N_TRIALS} trials) ===")
    study = optuna.create_study(
        study_name  = STUDY_NAME,
        storage     = STORAGE,
        direction   = "maximize",
        load_if_exists = True,
    )
    study.optimize(
        lambda trial: objective(trial, train_df, train_df_single, feature_cols, single_win_indices),
        n_trials    = N_TRIALS,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n  Best trial: #{best.number}")
    print(f"  Best score: {best.value:.6f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    best_params = {**best.params, "random_state": RANDOM_STATE}
    with open(MODELS_DIR / "best_params_xgb.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("  Saved: outputs/models/best_params_xgb.json")


if __name__ == "__main__":
    main()
