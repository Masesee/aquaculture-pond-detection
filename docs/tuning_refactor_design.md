# Design Specification: Unified Hyperparameter Tuning & Refactoring

This document defines the design and implementation details for the unified hyperparameter tuning pipeline of the XGBoost and CatBoost classifiers, ensuring alignment with the raw-probability cross-validation structure of the main LightGBM pipeline.

---

## 1. Understanding Summary

*   **What is being built:** A unified tuning infrastructure using a shared Out-of-Fold (OOF) CV loop utility (`tuning_utils.py`) to run hyperparameter sweeps for XGBoost (`tune_xgb.py`) and CatBoost (`tune_catboost.py`).
*   **Why it exists:** To align XGBoost and CatBoost tuning with the correct raw-probability fold mask mapping used by `train.py`, while eliminating duplicate loop and scoring logic to prevent code drift.
*   **Who it is for:** Zindi Aquaculture Pond Identification pipeline.
*   **Key constraints:**
    *   No post-hoc calibration or prior corrections (100% compliant with Zindi Rule 2).
    *   No inline duplication of cross-validation fold mapping or scoring logic.
*   **Explicit non-goals:**
    *   No Optuna trial-level pruning (complete, unpruned OOF arrays are required for downstream blend optimization).
    *   No trial resumption from stale database files (SQLite files must be cleared before tuning).

---

## 2. Assumptions & Benchmarks

*   **Fold Mask Mapping:** `get_fold_train_mask` from `cv_strategy.py` is the single source of truth for splitting training and validation folds.
*   **Tuning Durations (heuristic 1.3x scaling from midpoint benchmarks):**
    *   *XGBoost (100 trials):* Estimated at ~62s per trial $\rightarrow$ **~1.7 hours**.
    *   *CatBoost (60 trials):* Estimated at ~102s per trial $\rightarrow$ **~1.7 hours**.
    *   *Total Sequential Sweep:* **~3.4 hours**.
*   **Safety Monitoring:** Wall-clock elapsed time will be logged every 10 trials to detect timing overruns early.

---

## 3. Architecture & Components

```
pipelines/training/
├── cv_strategy.py      <-- Provides get_fold_train_mask()
├── tuning_utils.py     <-- Shared run_oof_tuning_loop()
├── tune_xgb.py         <-- XGBoost parameter space + factory
└── tune_catboost.py    <-- CatBoost parameter space + factory
```

### Shared CV Loop: `run_oof_tuning_loop`
```python
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
    # Samples parameters once per trial
    params = param_space_fn(trial)
    ...
    # Loops over folds, generates mask via cv_strategy, instantiates model_factory(params)
    ...
    # Computes combined_score(F1, AUC) on final OOF predictions
```

---

## 4. Decision Log

| Decision | Alternatives Considered | Rationale |
| :--- | :--- | :--- |
| **Two-Commit Sequence** | Single Monolithic Commit | Commit 1 isolates the refactoring of `tune_xgb.py` to verify zero functional OOF score regression. Commit 2 implements the general `tuning_utils.py` and the new CatBoost tuner, ensuring clear commit-level attribution. |
| **Functional Factory (`tuning_utils.py`)** | Abstract Base Class (ABC) | A functional parameter factory keeps exactly one code path for the cross-validation fold logic. It prevents subclass drift and complies with "vanilla by default", avoiding unnecessary class hierarchy abstractions. |
| **XGBoost ≥ 2.0 early-stopping API** | fit-level argument | In XGBoost 3.2.0, `early_stopping_rounds` must be set in the constructor (`XGBClassifier`) with validation data passed to `.fit()`. CatBoost accepts early stopping at `.fit()`. Aligning these ensures both get early stopping. |
| **Tuning Budgets (100 XGB / 60 CB)** | 100 Trials Each | CatBoost's search space is 4D (vs 8D for XGBoost) and converges faster. Reducing it to 60 trials keeps the estimated total tuning time under ~3.4 hours. Average trial times (~62s XGB / ~102s CB) are estimated via a 1.3x heuristic scaling factor on midpoint benchmarks and will be monitored dynamically during runs. |
| **No Trial-Level Pruning** | Optuna `MedianPruner` | Complete, unpruned OOF predictions are mandatory to evaluate stable ensembling weights downstream; partial OOF vectors would introduce scale mismatches. |
