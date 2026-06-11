"""
Base learner definitions for the stacking ensemble.

Three learners with deliberately different inductive biases:
  1. LightGBM    — nonlinear, captures complex feature interactions
  2. RandomForest — bagged trees, lower variance, different split heuristic
  3. LogisticRegression on top SHAP features — linear, high-bias anchor

Each learner exposes a consistent interface:
  fit(X, y) → self
  predict_proba(X) → np.ndarray shape (n, 2)

This module has no side effects and no file I/O.
"""

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ── LightGBM ──────────────────────────────────────────────────────────────────

# Best params from Optuna run. Hardcoded here so the ensemble
# does not depend on best_params.json existing on disk.
LGBM_BEST_PARAMS: dict = {
    "objective":          "binary",
    "boosting_type":      "gbdt",
    "learning_rate":      0.04897653110157577,
    "num_leaves":         79,
    "max_depth":          5,
    "min_child_samples":  30,
    "subsample":          0.6987328926522475,
    "subsample_freq":     1,
    "colsample_bytree":   0.9943288097877028,
    "reg_alpha":          0.034629363027917796,
    "reg_lambda":         0.15860861985219954,
    "class_weight":       None,
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
    # n_estimators set at fit time using early stopping
    "n_estimators":       1000,
}

EARLY_STOPPING_ROUNDS = 50


def make_lgbm(n_estimators: int | None = None) -> lgb.LGBMClassifier:
    """
    Returns a LightGBM classifier with best known params.
    n_estimators overrides the default when provided (used for final fit
    where early stopping is unavailable).
    """
    params = {**LGBM_BEST_PARAMS}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return lgb.LGBMClassifier(**params)


# ── Random Forest ─────────────────────────────────────────────────────────────

RF_PARAMS: dict = {
    "n_estimators":      500,
    "max_depth":         None,       # fully grown — bagging provides regularization
    "min_samples_leaf":  5,          # conservative for 963 samples
    "max_features":      "sqrt",     # standard for classification
    "class_weight":      None,       # consistent with LightGBM decision
    "random_state":      42,
    "n_jobs":            -1,
}


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


# ── Logistic Regression ───────────────────────────────────────────────────────

# Top SHAP features from the analysis — water physics signal only.
# Excludes raw band features that showed temporal sensitivity.
# This list is the invariant anchor: if water indices say "pond",
# LogReg says "pond" regardless of what the tree models learned.
LOGREG_FEATURES: list[str] = [
    "NDWI__max",
    "NDWI__p90",
    "NDWI__mean",
    "NDWI__min",
    "AWEInsh__cv",
    "AWEInsh__p90",
    "AWEInsh__mean",
    "MNDWI__cv",
    "MNDWI__p90",
    "MNDWI__mean",
    "NDVI__cv",
    "NDVI__min",
    "NDWI_pos_count",
    "MNDWI_pos_count",
    "AWEInsh_pos_count",
    "SAR_diff_db__mean",
    "swir1__cv",
    "VV__mean",
    "VH__median",
    "region",
    "dist_to_pond_centroid",
]

LOGREG_PARAMS: dict = {
    "C":            0.1,       # strong regularization — 21 features, 963 samples
    "max_iter":     1000,
    "solver":       "lbfgs",
    "class_weight": None,
    "random_state": 42,
}


def make_logreg() -> Pipeline:
    """
    Returns a Pipeline: StandardScaler → LogisticRegression.
    Scaler is mandatory — LogReg is sensitive to feature scale.
    The pipeline exposes fit/predict_proba directly.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**LOGREG_PARAMS)),
    ])