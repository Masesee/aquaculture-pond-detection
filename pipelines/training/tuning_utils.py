"""
Shared utilities for Out-of-Fold hyperparameter tuning.
Prevents duplicate fold construction and scoring logic.
"""
import time
from typing import Callable, Any, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score, roc_auc_score
from contracts.schema import TARGET_COL
from pipelines.training.cv_strategy import make_cv_splits, get_fold_train_mask
from pipelines.evaluation.metrics import combined_score


class TuningProgressCallback:
    """
    Optuna callback to log tuning progress, total elapsed time,
    and running average trial times every N trials.
    """
    def __init__(self, model_name: str, print_every: int = 10):
        self.model_name = model_name
        self.print_every = print_every
        self.start_time = time.time()
        
    def __call__(self, study: optuna.Study, trial: Any) -> None:
        trial_num = len(study.trials)
        if trial_num % self.print_every == 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / trial_num
            print(
                f"[{self.model_name}] Completed {trial_num} trials. "
                f"Elapsed: {elapsed:.1f}s | Avg: {avg_time:.1f}s/trial | "
                f"Best Score: {study.best_value:.6f}"
            )


def run_oof_tuning_loop(
    train_df: pd.DataFrame,
    train_df_single: pd.DataFrame,
    feature_cols: list[str],
    param_space_fn: Callable[[Any], dict],
    model_factory: Callable[[dict], Any],
    trial: Any,
    fit_predict_fn: Optional[Callable[[Any, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray], np.ndarray]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    Executes the 5-fold CV group split, trains models, collects OOF predictions,
    and returns the combined score (0.6*F1 + 0.4*AUC).
    
    param_space_fn(trial) -> dict: 
        Samples hyperparameters ONCE per trial via trial.suggest_*(...) calls.
        Called exactly once before the fold loop.
        
    model_factory(params) -> Any:
        Instantiates a fresh, untrained model from the fixed param dict.
        Called once per fold.
        
    Default fit_predict_fn expects standard sklearn contract:
        - calls model.fit(X_tr, y_tr)
        - returns model.predict_proba(X_val)[:, 1] (class-1 probability).
    
    For models requiring early stopping on eval_set (XGBoost, CatBoost),
    fit_predict_fn is mandatory to pass eval_set to fit() and return probabilities.
    """
    # Sample hyperparameters once before the loop
    params = param_space_fn(trial)
    
    splits = make_cv_splits(train_df_single, n_splits=n_splits, random_state=random_state)
    y_train_single = train_df_single[TARGET_COL].values
    oof_probs = np.zeros(len(train_df_single), dtype=float)
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])
    
    for fold, (train_pos, val_pos) in enumerate(splits):
        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        X_tr = train_df.loc[train_mask, feature_cols]
        y_tr = train_df.loc[train_mask, TARGET_COL].values
        
        X_val = train_df_single[feature_cols].iloc[val_pos]
        y_val = y_train_single[val_pos]
        
        # Instantiate a fresh model per fold using the fixed params
        model = model_factory(params)
        
        if fit_predict_fn is not None:
            fold_probs = fit_predict_fn(model, X_tr, y_tr, X_val, y_val)
        else:
            model.fit(X_tr, y_tr)
            fold_probs = model.predict_proba(X_val)[:, 1]
            
        oof_probs[val_pos] = fold_probs

    preds = (oof_probs >= 0.5).astype(int)
    f1 = f1_score(y_train_single, preds)
    auc = roc_auc_score(y_train_single, oof_probs)
    return combined_score(f1, auc)
