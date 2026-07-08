"""
CatBoost training pipeline.

Trains a CatBoost Classifier on the domain-invariant feature set using single-window
Stratified Group CV, saving OOF predictions and test probabilities for ensembling.

Run with:
    python -m pipelines.training.train_catboost
"""

import sys
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL, WINDOW_METADATA_COLS
from pipelines.training.cv_strategy import (
    make_cv_splits,
    get_single_window_indices,
)

PROCESSED_DIR  = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "outputs" / "models"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

CAT_PARAMS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 0,
}


def combined_score(f1: float, auc: float) -> float:
    return 0.6 * f1 + 0.4 * auc


def main() -> None:
    print("=== [CatBoost] Loading feature matrices ===")

    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    invariant_path = ROOT / "outputs" / "features" / "invariant_features.txt"
    if invariant_path.exists():
        with open(invariant_path) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(feature_cols)} robust features from invariant_features.txt")
    else:
        feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]

    exclude_metadata = "--exclude-metadata" in sys.argv
    metadata_cols = WINDOW_METADATA_COLS
    if exclude_metadata:
        feature_cols = [c for c in feature_cols if c not in metadata_cols]
        print(f"  Excluded window metadata. Remaining: {len(feature_cols)}")
    else:
        for col in metadata_cols:
            if col not in feature_cols and col in train_df.columns:
                feature_cols.append(col)
        print(f"  Included window metadata. Total features: {len(feature_cols)}")


    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]

    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)

    # ── Optional pseudo-labeling ───────────────────────────────────────────
    use_pseudo = "--pseudo" in sys.argv
    if use_pseudo:
        from pipelines.training.train import load_pseudo_labels
        sub_path = SUBMISSIONS_DIR / "blend_submission_best.csv"
        if not sub_path.exists():
            sub_path = SUBMISSIONS_DIR / "submission.csv"
        if not sub_path.exists():
            print(f"\nERROR: Pseudo-labeling requires a source submission at: {sub_path}")
            return
        print("\n=== [CatBoost] Loading pseudo-labels ===")
        pseudo_df, n_pseudo = load_pseudo_labels(
            submission_path    = sub_path,
            test_features_path = PROCESSED_DIR   / "test_features.parquet",
        )
        train_augmented = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"  Training size: {len(train_df)} -> {len(train_augmented)} (+{n_pseudo} pseudo-labeled)")
    else:
        train_augmented = train_df

    print(f"\n=== [CatBoost] {N_SPLITS}-fold stratified group CV (on 1-window subset) ===")
    splits = make_cv_splits(train_df_single, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    oof_probs   = np.zeros(len(train_df_single), dtype=float)
    fold_scores = []

    for fold, (train_pos, val_pos) in enumerate(splits):
        val_idx_in_train_df = single_win_indices[val_pos]
        val_base_ids = set(train_df_single.iloc[val_pos]["ID"].apply(lambda x: x.split("_w")[0]))

        X_val = train_df.iloc[val_idx_in_train_df][feature_cols]
        y_val = train_df.iloc[val_idx_in_train_df][TARGET_COL].values

        if use_pseudo:
            train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
            pseudo_mask = np.ones(len(pseudo_df), dtype=bool)
            full_train_mask = np.concatenate([train_mask, pseudo_mask])
            X_tr = train_augmented.loc[full_train_mask, feature_cols]
            y_tr = train_augmented.loc[full_train_mask, TARGET_COL].values
        else:
            train_mask = train_df["ID"].apply(lambda x: x.split("_w")[0] not in val_base_ids).values
            X_tr = train_df.loc[train_mask, feature_cols]
            y_tr = train_df.loc[train_mask, TARGET_COL].values

        model = cb.CatBoostClassifier(**CAT_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )

        fold_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_pos] = fold_probs

        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_f1    = f1_score(y_val, fold_preds)
        fold_auc   = roc_auc_score(y_val, fold_probs)
        fold_score = combined_score(fold_f1, fold_auc)

        fold_scores.append({"fold": fold, "f1": round(fold_f1, 4), "auc": round(fold_auc, 4), "score": round(fold_score, 4)})
        print(f"  Fold {fold}: F1={fold_f1:.4f} | AUC={fold_auc:.4f} | Score={fold_score:.4f}")

    y_train_single = train_df_single[TARGET_COL].values
    oof_preds = (oof_probs >= 0.5).astype(int)
    oof_f1    = f1_score(y_train_single, oof_preds)
    oof_auc   = roc_auc_score(y_train_single, oof_probs)
    oof_score = combined_score(oof_f1, oof_auc)
    print(f"\n  [CatBoost] OOF aggregate (pre-cal) — F1={oof_f1:.4f} | AUC={oof_auc:.4f} | Score={oof_score:.4f}")


    seeds = [42, 100, 2026]
    if "--seeds" in sys.argv:
        try:
            idx = sys.argv.index("--seeds") + 1
            seeds = [int(x) for x in sys.argv[idx].split(",")]
        except (ValueError, IndexError):
            pass
    print(f"\n=== [CatBoost] Training final model on full training data (Seed Averaging across {seeds}) ===")
    raw_test_probs_list = []
    
    for seed in seeds:
        seed_params = {**CAT_PARAMS, "random_seed": seed}
        model = cb.CatBoostClassifier(**seed_params)
        if use_pseudo:
            model.fit(train_augmented[feature_cols], train_augmented[TARGET_COL].values)
        else:
            model.fit(X_train, y_train)
        raw_test_probs_list.append(model.predict_proba(X_test)[:, 1])
        if seed == 42:
            joblib.dump(model, MODELS_DIR / "cb_model.joblib")
            
    raw_test_probs = np.mean(raw_test_probs_list, axis=0)

    oof_df = pd.DataFrame({
        "ID": train_df_single["ID"].values,
        "label": y_train_single,
        "oof_prob_raw": oof_probs,
    })
    oof_df.to_csv(MODELS_DIR / "cb_oof_predictions.csv", index=False)

    test_probs_df = pd.DataFrame({
        "ID": test_df["ID"].values,
        "cb_prob_raw": raw_test_probs,
    })
    test_probs_df.to_csv(MODELS_DIR / "cb_test_probs.csv", index=False)

    print("  Saved: outputs/models/cb_model.joblib (seed 42)")
    print("  Saved: outputs/models/cb_oof_predictions.csv")
    print("  Saved: outputs/models/cb_test_probs.csv")


if __name__ == "__main__":
    main()
