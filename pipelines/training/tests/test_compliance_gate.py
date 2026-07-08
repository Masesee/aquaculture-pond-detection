import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pipelines.training.blending import blend_raw_probs
from pipelines.training.blend_config import load_blend_weights

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "outputs" / "models"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"


def test_submitted_probs_equal_raw_blend_no_reshaping():
    """Regression gate: submission probabilities must equal the raw
    blended model output exactly. No isotonic remap, no prior shift.
    Protects Zindi Rule 2 compliance."""
    sub_path = SUBMISSIONS_DIR / "submission.csv"
    if not sub_path.exists():
        # Skip if submission file hasn't been generated yet (cold run)
        return

    lgbm_test = pd.read_csv(MODELS_DIR / "lgbm_test_probs.csv")
    xgb_test = pd.read_csv(MODELS_DIR / "xgb_test_probs.csv")
    cb_test = pd.read_csv(MODELS_DIR / "cb_test_probs.csv")
    sub = pd.read_csv(sub_path)

    raw_lgbm = lgbm_test["lgbm_prob_raw"].values
    raw_xgb = xgb_test["xgb_prob_raw"].values
    raw_cb = cb_test["cb_prob_raw"].values

    # Load ensembled weights dynamically from config source of truth
    weights = load_blend_weights()

    raw_blend = blend_raw_probs(raw_lgbm, raw_xgb, raw_cb, weights=weights)
    submitted = sub["TargetRAUC"].values

    np.testing.assert_array_almost_equal(raw_blend, submitted, decimal=6)


def test_no_calibration_code_in_pipelines():
    """Asserts that no active references to calibration code remain in
    non-test Python files under pipelines/training/."""

    
    # Search for forbidden terms
    forbidden_terms = ["IsotonicRegression", "correct_prior", "fit_calibrator"]
    
    for term in forbidden_terms:
        # Run grep command to find matches
        try:
            result = subprocess.run(
                ["git", "grep", "-n", term, "--", "*.py", ":(exclude)*tests*"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            # If matches found, fail the test
            if result.stdout:
                assert False, f"Forbidden term '{term}' found in non-test codebase:\n{result.stdout}"
        except subprocess.CalledProcessError:
            # git grep exits with 1 if no matches are found, which is what we want!
            pass


def test_no_hardcoded_blend_weights():
    """Asserts that the current optimal weights are not hardcoded inside
    any production model or ensembling scripts."""
    try:
        weights = load_blend_weights()
    except FileNotFoundError:
        return

    # Check for hardcoded occurrences of these specific float values (formatted to 3 decimals)
    for w in weights:
        # Avoid checking 1/3 or equal weights defaults
        if np.isclose(w, 1/3, atol=1e-3):
            continue
            
        w_str = f"{w:.3f}"
        
        # Git grep for the weight literal, excluding tests, config, optimizer, docs, and scratch files
        try:
            result = subprocess.run(
                [
                    "git", "grep", "-n", "-F", w_str, "--", "*.py", 
                    ":(exclude)*tests*", 
                    ":(exclude)*blend_config.py", 
                    ":(exclude)*optimize_blend.py"
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                assert False, f"Hardcoded weight '{w_str}' found in codebase:\n{result.stdout}"
        except subprocess.CalledProcessError:
            pass


def test_no_group_leakage_across_folds():
    """Asserts that for every fold, the base IDs in the validation fold
    are completely disjoint from the base IDs in the training fold.
    Verifies that get_fold_train_mask successfully prevents leakage
    under group cross-validation."""
    from pipelines.training.cv_strategy import make_cv_splits, get_single_window_indices, get_fold_train_mask
    
    train_path = ROOT / "data" / "processed" / "train_features.parquet"
    if not train_path.exists():
        # Skip if features haven't been processed yet
        return

    train_df = pd.read_parquet(train_path)
    single_win_indices = get_single_window_indices(train_df, random_state=42)
    train_df_single = train_df.iloc[single_win_indices].reset_index(drop=True)
    base_ids_full = train_df["ID"].apply(lambda x: x.split("_w")[0])

    splits = make_cv_splits(train_df_single, n_splits=5, random_state=42)

    for fold_idx, (train_pos, val_pos) in enumerate(splits):
        # Validation base IDs
        val_base_ids = set(train_df_single.iloc[val_pos]["ID"].apply(lambda x: x.split("_w")[0]))
        
        # Training mask mapped to full train_df
        train_mask = get_fold_train_mask(train_df, train_df_single, val_pos, base_ids_full)
        train_base_ids = set(train_df.loc[train_mask, "ID"].apply(lambda x: x.split("_w")[0]))
        
        # Assert intersection is empty
        leakage = val_base_ids.intersection(train_base_ids)
        assert len(leakage) == 0, f"Fold {fold_idx} has group leakage: {leakage}"


