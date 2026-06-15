"""
Hard case analysis.

Loads OOF predictions, identifies training samples the model consistently
gets wrong, and computes feature-level statistics to find discriminating patterns.

Run with:
    python -m pipelines.evaluation.hard_case_analysis
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR     = ROOT / "outputs" / "models"
PROCESSED_DIR  = ROOT / "data"    / "processed"
EVAL_DIR       = ROOT / "outputs" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=== Loading OOF predictions and features ===")
    oof = pd.read_csv(MODELS_DIR / "oof_predictions.csv")
    train = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")

    feature_cols = [c for c in train.columns if c not in ["ID", "label"]]
    merged = train.merge(oof[["ID", "oof_prob_cal", "oof_pred"]], on="ID", how="left")

    # ── Classify OOF outcomes ──────────────────────────────────────────────────
    y = merged["label"].values
    y_pred = merged["oof_pred"].values
    prob = merged["oof_prob_cal"].values

    tp = (y == 1) & (y_pred == 1)
    tn = (y == 0) & (y_pred == 0)
    fp = (y == 0) & (y_pred == 1)   # false positive: predicted pond, actually not
    fn = (y == 1) & (y_pred == 0)   # false negative: predicted non-pond, actually pond

    print(f"  TP={tp.sum()} | TN={tn.sum()} | FP={fp.sum()} | FN={fn.sum()}")
    print(f"  Total wrong: {(fp | fn).sum()}")

    # ── Probability margin analysis ────────────────────────────────────────────
    # Hard cases: wrong AND probability near 0.5 (genuinely ambiguous)
    # vs wrong AND probability far from 0.5 (model was confidently wrong)
    wrong_mask = fp | fn
    wrong_df = merged[wrong_mask].copy()
    wrong_df["error_type"] = np.where(fp[wrong_mask], "FP", "FN")
    wrong_df["prob_margin"] = (wrong_df["oof_prob_cal"] - 0.5).abs()

    print("\n=== Wrong prediction breakdown ===")
    print(wrong_df[["ID", "label", "oof_prob_cal", "oof_pred", "error_type", "prob_margin"]]
          .sort_values("prob_margin")
          .to_string(index=False))

    # ── Feature statistics comparison ─────────────────────────────────────────
    print("\n=== Top features where hard cases differ from correct predictions ===")

    hard_idx   = np.where(wrong_mask)[0]
    correct_idx = np.where(~wrong_mask)[0]

    diffs = {}
    for col in feature_cols:
        hard_mean    = merged[col].iloc[hard_idx].mean()
        correct_mean = merged[col].iloc[correct_idx].mean()
        hard_std     = merged[col].iloc[hard_idx].std()
        correct_std  = merged[col].iloc[correct_idx].std()
        # Normalized difference: how many pooled SDs apart?
        pooled_std = np.sqrt((hard_std**2 + correct_std**2) / 2 + 1e-9)
        diffs[col] = abs(hard_mean - correct_mean) / pooled_std

    diff_df = pd.Series(diffs).sort_values(ascending=False).reset_index()
    diff_df.columns = ["feature", "normalized_diff"]
    print(diff_df.head(30).to_string(index=False))

    # ── FP vs FN feature comparison ───────────────────────────────────────────
    print("\n=== FP vs FN: key feature differences ===")
    fp_df = merged[fp]
    fn_df = merged[fn]

    key_features = [
        "NDWI__max", "NDTI__min", "NDWI2__std", "SAR_RVI__mean",
        "SABI__cv", "AWEInsh__cv", "MNDWI__cv", "VV__mean",
        "region", "water_index_unanimous",
    ]
    # Filter to only features that exist in current feature set
    key_features = [f for f in key_features if f in feature_cols]

    comparison = pd.DataFrame({
        "feature":  key_features,
        "FP_mean":  [fp_df[f].mean()  for f in key_features],
        "FN_mean":  [fn_df[f].mean()  for f in key_features],
        "TP_mean":  [merged[tp][f].mean() for f in key_features],
        "TN_mean":  [merged[tn][f].mean() for f in key_features],
    })
    print(comparison.to_string(index=False))

    # ── Margin histogram ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(prob[tp], bins=20, alpha=0.6, label="TP", color="green")
    axes[0].hist(prob[fn], bins=10, alpha=0.7, label="FN", color="red")
    axes[0].axvline(0.5, color="black", linestyle="--")
    axes[0].set_title("Ponds: correct (TP) vs missed (FN)")
    axes[0].set_xlabel("Calibrated probability")
    axes[0].legend()

    axes[1].hist(prob[tn], bins=20, alpha=0.6, label="TN", color="blue")
    axes[1].hist(prob[fp], bins=10, alpha=0.7, label="FP", color="orange")
    axes[1].axvline(0.5, color="black", linestyle="--")
    axes[1].set_title("Non-ponds: correct (TN) vs false alarms (FP)")
    axes[1].set_xlabel("Calibrated probability")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(EVAL_DIR / "hard_case_prob_dist.png", dpi=150)
    print(f"\n  Saved: {EVAL_DIR / 'hard_case_prob_dist.png'}")

    # ── Save hard case feature matrix ─────────────────────────────────────────
    wrong_features = merged[wrong_mask][["ID", "label", "oof_prob_cal", "oof_pred", "prob_margin"] + feature_cols]
    wrong_features.to_csv(EVAL_DIR / "hard_cases_features.csv", index=False)
    print(f"  Saved: {EVAL_DIR / 'hard_cases_features.csv'}")

    # ── Region breakdown of errors ────────────────────────────────────────────
    print("\n=== Error breakdown by region ===")
    for region in [0, 1]:
        r_mask = merged["region"] == region
        r_fp = (fp & r_mask).sum()
        r_fn = (fn & r_mask).sum()
        r_total = r_mask.sum()
        print(f"  Region {region} (n={r_total}): FP={r_fp} | FN={r_fn} | "
              f"error rate={100*(r_fp+r_fn)/r_total:.1f}%")

    print("\n=== Hard case analysis complete ===")


if __name__ == "__main__":
    main()
