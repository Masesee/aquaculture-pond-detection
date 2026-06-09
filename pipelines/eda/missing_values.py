"""
Q3: Are there missing values? Do they cluster by month or by location?
Outputs:
  - missing_train.csv
  - missing_heatmap_train.png  (bands × months)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contracts.schema import ALL_BANDS, MONTHS, raw_col


def run_missing_values(
    train: pd.DataFrame, test: pd.DataFrame, out_dir: Path
) -> None:
    for name, df in [("train", train), ("test", test)]:
        feature_cols = [raw_col(b, m) for b in ALL_BANDS for m in MONTHS]
        present_cols = [c for c in feature_cols if c in df.columns]

        missing = df[present_cols].isnull().sum()
        total_missing = missing.sum()
        print(f"  [{name}] Total missing cells: {total_missing} / {df.shape[0] * len(present_cols)}")

        if total_missing == 0:
            print(f"  [{name}] No missing values.")
            pd.DataFrame({"missing_count": missing}).to_csv(
                out_dir / f"missing_{name}.csv"
            )
            continue

        # Save raw counts
        missing_df = pd.DataFrame({"missing_count": missing})
        missing_df.to_csv(out_dir / f"missing_{name}.csv")

        # Reshape to bands × months heatmap
        matrix = np.zeros((len(ALL_BANDS), len(MONTHS)), dtype=int)
        for i, band in enumerate(ALL_BANDS):
            for j, month in enumerate(MONTHS):
                col = raw_col(band, month)
                if col in df.columns:
                    matrix[i, j] = df[col].isnull().sum()

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            matrix,
            xticklabels=MONTHS,
            yticklabels=ALL_BANDS,
            cmap="Reds",
            annot=True,
            fmt="d",
            linewidths=0.3,
            ax=ax,
        )
        ax.set_title(f"Missing Values — {name} (bands × months)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Band")
        plt.tight_layout()
        fig.savefig(out_dir / f"missing_heatmap_{name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: missing_heatmap_{name}.png")