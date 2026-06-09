"""
Optuna hyperparameter sweep for LightGBM.
Optimises the Zindi combined score (0.6*F1 + 0.4*AUC) on OOF predictions.

Run with:
    python -m pipelines.training.tune

Outputs:
    experiments/logs/optuna_study.db       ← resumable SQLite study
    experiments/logs/optuna_results.csv    ← all trial results
    outputs/models/best_params.json        ← best hyperparameters
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
from pipelines.training.cv_strategy import make_cv_splits
from pipelines.evaluation.metrics import combined_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR  = ROOT / "data"        / "processed"
LOGS_DIR       = ROOT / "experiments" / "logs"
MODELS_DIR     = ROOT / "outputs"     / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS      = 5
RANDOM_STATE  = 42
N_TRIALS      = 100
STUDY_NAME    = "lgbm_aquaculture"
STORAGE       = f"sqlite:///{LOGS_DIR / 'optuna_study.db'}"

EARLY_STOPPING_ROUNDS = 50


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
    params = {
        "objective":         "binary",
        "boosting_type":     "gbdt",
        "n_estimators":      1000,
        "learning_rate":     trial.suggest_float("learning_rate",  0.005, 0.1,  log=True),
        "num_leaves":        trial.suggest_int(  "num_leaves",     16,    128),
        "max_depth":         trial.suggest_int(  "max_depth",      3,     10),
        "min_child_samples": trial.suggest_int(  "min_child_samples", 5,  50),
        "subsample":         trial.suggest_float("subsample",      0.5,   1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",      1e-3,  10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",     1e-3,  10.0, log=True),
        "class_weight":      "balanced",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    splits    = make_cv_splits(
        pd.DataFrame({"label": y, "region": X["region"].values}),
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )
    oof_probs = np.zeros(len(y), dtype=float)

    for train_pos, val_pos in splits:
        X_tr  = X.iloc[train_pos]
        y_tr  = y[train_pos]
        X_val = X.iloc[val_pos]
        y_val = y[val_pos]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        oof_probs[val_pos] = model.predict_proba(X_val)[:, 1]

    preds = (oof_probs >= 0.5).astype(int)
    f1    = f1_score(y, preds)
    auc   = roc_auc_score(y, oof_probs)
    return combined_score(f1, auc)


def main() -> None:
    print("=== Loading features ===")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values
    print(f"  X: {X.shape} | positive rate: {y.mean():.3f}")

    print(f"\n=== Running Optuna ({N_TRIALS} trials) ===")
    study = optuna.create_study(
        study_name  = STUDY_NAME,
        storage     = STORAGE,
        direction   = "maximize",
        load_if_exists = True,
    )
    study.optimize(
        lambda trial: objective(trial, X, y),
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