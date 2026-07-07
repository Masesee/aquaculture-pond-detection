"""
Optuna hyperparameter sweep for LightGBM on masked temporal features.
Optimises the Zindi combined score (0.6*F1 + 0.4*AUC) on OOF predictions
using StratifiedGroupKFold to prevent data leakage.

Run with:
    python -m pipelines.training.tune

Outputs:
    experiments/logs/optuna_study_masked.db       ← fresh resumable SQLite study
    experiments/logs/optuna_results.csv           ← all trial results
    outputs/models/best_params.json               ← best hyperparameters
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL
import functools
from pipelines.training.cv_strategy import make_cv_splits, get_single_window_indices, get_fold_train_mask
from pipelines.evaluation.metrics import combined_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR  = ROOT / "data"        / "processed"
LOGS_DIR       = ROOT / "experiments" / "logs"
MODELS_DIR     = ROOT / "outputs"     / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS      = 5
RANDOM_STATE  = 42
N_TRIALS      = 100
STUDY_NAME    = "lgbm_aquaculture_regularized"
STORAGE       = f"sqlite:///{LOGS_DIR / 'optuna_study_regularized.db'}"


def objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    single_win_indices: np.ndarray,
    train_df_single: pd.DataFrame,
    base_ids_full: pd.Series,
) -> float:
    # Tuned parameters space search centered around robust baseline settings
    params = {
        "objective":         "binary",
        "boosting_type":     "gbdt",
        "n_estimators":      trial.suggest_int(  "n_estimators",      150,    450),
        "learning_rate":     trial.suggest_float("learning_rate",      0.01,   0.08, log=True),
        "num_leaves":        trial.suggest_int(  "num_leaves",         15,     45),
        "max_depth":         trial.suggest_int(  "max_depth",          3,      6),
        "min_child_samples": trial.suggest_int(  "min_child_samples",  100,    300),
        "subsample":         trial.suggest_float("subsample",          0.60,   0.90),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree",   0.30,   0.70),
        "reg_alpha":         trial.suggest_float("reg_alpha",          0.01,   10.0,  log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",         0.1,    50.0,  log=True),
        "class_weight":      None,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    y_single = train_df_single[TARGET_COL].values
    oof_probs = np.zeros(len(y_single), dtype=float)

    for train_pos, val_pos in splits:
        val_idx_in_train_df = single_win_indices[val_pos]

        X_val = train_df.iloc[val_idx_in_train_df][feature_cols]

        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        oof_probs[val_pos] = model.predict_proba(X_val)[:, 1]

    preds = (oof_probs >= 0.5).astype(int)
    f1    = f1_score(y_single, preds)
    auc   = roc_auc_score(y_single, oof_probs)
    return combined_score(f1, auc)


def main() -> None:
    print("=== Loading features ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    
    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    if invariant_path.exists():
        with open(invariant_path) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(feature_cols)} robust features from invariant_features.txt")
    else:
        feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]
        print(f"  Using all {len(feature_cols)} features (no invariant_features.txt found)")
    print(f"  X: {train_df[feature_cols].shape} | positive rate: {train_df[TARGET_COL].mean():.3f}")

    # Prepare hoisted splits and indices once
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])
    single_win_indices = get_single_window_indices(train_df, random_state=RANDOM_STATE)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)
    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    # Bind variables to objective using partial
    obj_func = functools.partial(
        objective,
        train_df=train_df,
        feature_cols=feature_cols,
        splits=splits,
        single_win_indices=single_win_indices,
        train_df_single=train_df_single,
        base_ids_full=base_ids_full
    )

    print(f"\n=== Running Optuna ({N_TRIALS} trials) ===")
    study = optuna.create_study(
        study_name  = STUDY_NAME,
        storage     = STORAGE,
        direction   = "maximize",
        load_if_exists = True,
    )
    study.optimize(
        obj_func,
        n_trials    = N_TRIALS,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n  Best trial: #{best.number}")
    print(f"  Best score: {best.value:.6f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(LOGS_DIR / "optuna_results.csv", index=False)

    best_params = {**best.params, "random_state": RANDOM_STATE}
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("\n  Saved: experiments/logs/optuna_results.csv")
    print("  Saved: outputs/models/best_params.json")
    print("\n=== Tuning complete. Run train.py to retrain with best params. ===")


if __name__ == "__main__":
    main()