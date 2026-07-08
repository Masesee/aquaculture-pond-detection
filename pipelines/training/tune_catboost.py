"""
Optuna hyperparameter sweep for CatBoost on masked temporal features.
Optimises the Zindi combined score (0.6*F1 + 0.4*AUC) on OOF predictions
using StratifiedGroupKFold to prevent data leakage.

Run with:
    python -m pipelines.training.tune_catboost
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import optuna
import catboost as cb

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
N_TRIALS      = 60
STUDY_NAME    = "cb_aquaculture_regularized"
STORAGE       = f"sqlite:///{LOGS_DIR / 'optuna_study_cb.db'}"


def suggest_cb_params(trial: optuna.Trial) -> dict:
    """Samples hyperparameters once per trial."""
    return {
        "iterations":        trial.suggest_int(  "iterations",        150,    450),
        "learning_rate":     trial.suggest_float("learning_rate",     0.02,   0.08, log=True),
        "depth":             trial.suggest_int(  "depth",             3,      6),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg",       1.0,    10.0),
        "random_seed":       RANDOM_STATE,
        "verbose":           0,
        "thread_count":      -1,
    }


def create_cb_model(params: dict) -> cb.CatBoostClassifier:
    """Instantiates a fresh model from the parameter dictionary."""
    return cb.CatBoostClassifier(**params)


def fit_predict_cb(model: cb.CatBoostClassifier, X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> np.ndarray:
    """Trains model with early stopping and returns class-1 probabilities."""
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    return model.predict_proba(X_val)[:, 1]


def objective(trial: optuna.Trial, train_df: pd.DataFrame, train_df_single: pd.DataFrame, feature_cols: list[str]) -> float:
    """Optuna objective function delegating to the shared loop."""
    return run_oof_tuning_loop(
        train_df=train_df,
        train_df_single=train_df_single,
        feature_cols=feature_cols,
        param_space_fn=suggest_cb_params,
        model_factory=create_cb_model,
        trial=trial,
        fit_predict_fn=fit_predict_cb,
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

    print(f"\n=== Running Optuna for CatBoost ({N_TRIALS} trials) ===")
    study = optuna.create_study(
        study_name  = STUDY_NAME,
        storage     = STORAGE,
        direction   = "maximize",
        load_if_exists = True,
    )
    
    progress_callback = TuningProgressCallback("CatBoost", print_every=10)
    
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

    best_params = {**best.params, "random_seed": RANDOM_STATE}
    with open(MODELS_DIR / "best_params_cb.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("  Saved: outputs/models/best_params_cb.json")


if __name__ == "__main__":
    main()
