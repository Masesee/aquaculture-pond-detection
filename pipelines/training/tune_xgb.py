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

from contracts.schema import WINDOW_METADATA_COLS
from pipelines.training.cv_strategy import get_single_window_indices
from pipelines.training.tuning_utils import run_oof_tuning_loop, TuningProgressCallback

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR  = ROOT / "data"        / "processed"
LOGS_DIR       = ROOT / "experiments" / "logs"
MODELS_DIR     = ROOT / "outputs"     / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS      = 5
RANDOM_STATE  = 42
N_TRIALS      = 100
STUDY_NAME    = "xgb_aquaculture_regularized"
STORAGE       = f"sqlite:///{LOGS_DIR / 'optuna_study_xgb.db'}"


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Samples hyperparameters once per trial."""
    return {
        "n_estimators":      trial.suggest_int(  "n_estimators",      150,    450),
        "learning_rate":     trial.suggest_float("learning_rate",      0.02,   0.08, log=True),
        "max_depth":         trial.suggest_int(  "max_depth",          3,      9),
        "min_child_weight":  trial.suggest_int(  "min_child_weight",   1,      20),
        "subsample":         trial.suggest_float("subsample",          0.50,   0.95),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",   0.20,   0.80),
        "reg_alpha":         trial.suggest_float("reg_alpha",          0.01,   10.0,  log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",         0.1,    30.0,  log=True),
        "eval_metric":       "logloss",
        "n_jobs":            -1,
        "random_state":      RANDOM_STATE,
    }


def create_xgb_model(params: dict) -> xgb.XGBClassifier:
    """Instantiates a fresh model from the parameter dictionary."""
    return xgb.XGBClassifier(**params, early_stopping_rounds=50)


def fit_predict_xgb(model: xgb.XGBClassifier, X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> np.ndarray:
    """Trains model with early stopping and returns class-1 probabilities."""
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model.predict_proba(X_val)[:, 1]


def objective(trial: optuna.Trial, train_df: pd.DataFrame, train_df_single: pd.DataFrame, feature_cols: list[str]) -> float:
    """Optuna objective function delegating to the shared loop."""
    return run_oof_tuning_loop(
        train_df=train_df,
        train_df_single=train_df_single,
        feature_cols=feature_cols,
        param_space_fn=suggest_xgb_params,
        model_factory=create_xgb_model,
        trial=trial,
        fit_predict_fn=fit_predict_xgb,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )


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
    
    progress_callback = TuningProgressCallback("XGBoost", print_every=10)
    
    study.optimize(
        lambda trial: objective(trial, train_df, train_df_single, feature_cols),
        n_trials    = N_TRIALS,
        callbacks   = [progress_callback],
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
