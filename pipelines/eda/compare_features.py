"""
Compare feature distributions between train and test datasets.
Run with:
    python -m pipelines.eda.compare_features
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contracts.schema import TARGET_COL

def main() -> None:
    PROCESSED_DIR = ROOT / "data" / "processed"
    
    train_path = PROCESSED_DIR / "train_features.parquet"
    test_path = PROCESSED_DIR / "test_features.parquet"
    
    if not train_path.exists() or not test_path.exists():
        print("ERROR: Processed feature matrices not found. Please run feature builder first.")
        sys.exit(1)
        
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    feature_cols = [c for c in test_df.columns if c not in ["ID", TARGET_COL]]

    rows = []
    for col in feature_cols:
        tr_mean = train_df[col].mean()
        tr_std  = train_df[col].std()
        te_mean = test_df[col].mean()
        te_std  = test_df[col].std()
        
        # Normalized difference in means
        mean_diff = np.abs(tr_mean - te_mean)
        norm_diff = mean_diff / (tr_std + 1e-9)
        
        rows.append({
            "feature": col,
            "train_mean": round(tr_mean, 4),
            "test_mean": round(te_mean, 4),
            "train_std": round(tr_std, 4),
            "test_std": round(te_std, 4),
            "norm_diff": round(norm_diff, 4)
        })

    diff_df = pd.DataFrame(rows).sort_values("norm_diff", ascending=False).reset_index(drop=True)

    print("=== Top 30 Features with the Largest Distribution Shift (Normalized Difference) ===")
    print(diff_df.head(30).to_string(index=False))

    # Also calculate if there is a shift in window metadata
    print("\n=== Window Metadata Comparison ===")
    for col in ["window_start", "window_length", "window_center"]:
        print(f"\nFeature: {col}")
        print("Train (augmented):")
        print(train_df[col].value_counts(normalize=True).sort_index().round(3))
        print("Test:")
        print(test_df[col].value_counts(normalize=True).sort_index().round(3))

if __name__ == "__main__":
    main()
