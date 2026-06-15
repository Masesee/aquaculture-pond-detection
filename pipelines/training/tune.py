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
N_TRIALS      = 200
# v6.2 study: fresh name for quarter-aggregation feature set (287 features).
# v6 DB (optuna_study_v6.db) was contaminated: mixed 256 + 247 feature trials.
# Always create a new DB per feature version to keep trial history clean.
STUDY_NAME    = "lgbm_aquaculture_v6_2"
STORAGE       = f"sqlite:///{LOGS_DIR / 'optuna_study_v6_2.db'}"


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
    # Sub 22 best: n_est=870, lr=0.1076, leaves=102, depth=8,
    #              min_child=85, subsample=0.647, colsample=0.373,
    #              reg_alpha=1.3e-4, reg_lambda=1.3e-3
    params = {
        "objective":         "binary",
        "boosting_type":     "gbdt",
        "n_estimators":      trial.suggest_int(  "n_estimators",      750,    980),
        "learning_rate":     trial.suggest_float("learning_rate",      0.085,  0.130, log=True),
        "num_leaves":        trial.suggest_int(  "num_leaves",         85,     120),
        "max_depth":         trial.suggest_int(  "max_depth",          6,      10),
        "min_child_samples": trial.suggest_int(  "min_child_samples",  70,     100),
        "subsample":         trial.suggest_float("subsample",          0.55,   0.75),
        "subsample_freq":    1,
        # v6.2: 295 features. Target ~76 features/tree (same as Sub 22).
        # 76/295 = 0.258. Range [0.22, 0.36] -> 65-106 features/tree.
        # Do NOT go above 0.38: at 295 feat that gives 112/tree -> pond over-prediction.
        "colsample_bytree":  trial.suggest_float("colsample_bytree",   0.22,   0.36),
        "reg_alpha":         trial.suggest_float("reg_alpha",          5e-5,   5e-4,  log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",         4e-4,   5e-3,  log=True),
        # class_weight=None matches train.py exactly — eliminates the
        # tuning/training objective inconsistency present since Sub 22.
        "class_weight":      None,
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

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
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
        study_name     = STUDY_NAME,
        storage        = STORAGE,
        direction      = "maximize",
        load_if_exists = True,   # safe: v6 DB is a fresh file
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