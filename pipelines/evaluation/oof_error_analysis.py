"""
OOF Error Analysis.

Loads the ensembled OOF predictions, identifies False Positives (FP) and
False Negatives (FN), joins them with their window metadata and physical
features, and prints out a detailed diagnostic report.

Run with:
    python -m pipelines.evaluation.oof_error_analysis
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

MODELS_DIR    = ROOT / "outputs" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    print("=== Loading OOF predictions and features ===")
    
    # 1. Load OOF predictions from all three models
    lgbm_oof = pd.read_csv(MODELS_DIR / "oof_predictions.csv")
    xgb_oof  = pd.read_csv(MODELS_DIR / "xgb_oof_predictions.csv")
    cb_oof   = pd.read_csv(MODELS_DIR / "cb_oof_predictions.csv")
    
    # Ensure they are aligned on ID
    xgb_oof = xgb_oof.set_index("ID").reindex(lgbm_oof["ID"]).reset_index()
    cb_oof  = cb_oof.set_index("ID").reindex(lgbm_oof["ID"]).reset_index()
    
    # Compute blended probabilities (equal weights)
    lgbm_probs = lgbm_oof["oof_prob_cal"].values
    xgb_probs  = xgb_oof["oof_prob_cal"].values
    cb_probs   = cb_oof["oof_prob_cal"].values
    blend_probs = (lgbm_probs + xgb_probs + cb_probs) / 3.0
    
    # Create aligned dataframe
    oof_df = pd.DataFrame({
        "ID": lgbm_oof["ID"],
        "label": lgbm_oof["label"],
        "pred_prob": blend_probs,
        "pred_label": (blend_probs >= 0.5).astype(int)
    })
    
    # Identify errors
    oof_df["error_type"] = "correct"
    oof_df.loc[(oof_df["label"] == 0) & (oof_df["pred_label"] == 1), "error_type"] = "FP"
    oof_df.loc[(oof_df["label"] == 1) & (oof_df["pred_label"] == 0), "error_type"] = "FN"
    
    # 2. Join with features (for window_start, window_length, raw band aggregates)
    train_feats = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    merged = oof_df.merge(train_feats, on="ID", how="inner")
    
    print(f"Total samples: {len(merged)}")
    print(f"Correct: {sum(merged['error_type'] == 'correct')}")
    print(f"False Positives (FP): {sum(merged['error_type'] == 'FP')}")
    print(f"False Negatives (FN): {sum(merged['error_type'] == 'FN')}")
    
    # 3. Analyze window parameters
    print("\n=== Window Parameter Analysis ===")
    for metric in ["window_length", "window_start"]:
        print(f"\nDistribution of {metric}:")
        counts = merged.groupby(["error_type", metric]).size().unstack(fill_value=0)
        # Normalize by total of that error type to get percentage
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        print("Counts:")
        print(counts.to_string())
        print("\nPercentages (%):")
        print(percentages.round(1).to_string())
        
    # 4. Analyze key index distributions (NDWI, VV, MNDWI, NDTI)
    print("\n=== Physical Feature Diagnostics ===")
    features_to_check = [
        "VV__mean", "VV__min", "VV__max", 
        "NDWI__max", "NDWI__mean",
        "MNDWI__max", "AWEInsh__max",
        "NDTI__max", "NDTI__min",
        "NDVI__min", "SWI__min"
    ]
    
    # Print mean value of these features for Correct, FP, and FN
    feat_summary = []
    for feat in features_to_check:
        if feat in merged.columns:
            group_means = merged.groupby("error_type")[feat].mean()
            feat_summary.append({
                "Feature": feat,
                "Correct_mean": group_means.get("correct", np.nan),
                "FP_mean": group_means.get("FP", np.nan),
                "FN_mean": group_means.get("FN", np.nan),
            })
            
    print(pd.DataFrame(feat_summary).to_string(index=False))

    # 5. Inspect individual hard cases (top incorrect predictions)
    print("\n=== Top 10 Worst False Positives (High probability but actually label=0) ===")
    fps = merged[merged["error_type"] == "FP"].sort_values("pred_prob", ascending=False)
    print(fps[["ID", "pred_prob", "window_start", "window_length", "VV__mean", "NDWI__max"]].head(10).to_string(index=False))

    print("\n=== Top 10 Worst False Negatives (Low probability but actually label=1) ===")
    fns = merged[merged["error_type"] == "FN"].sort_values("pred_prob", ascending=True)
    print(fns[["ID", "pred_prob", "window_start", "window_length", "VV__mean", "NDWI__max"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
