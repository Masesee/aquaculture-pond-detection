"""
Q1: What is the actual class balance?
Outputs:
  - class_balance.csv
  - class_balance.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from contracts.schema import TARGET_COL


def run_class_balance(train: pd.DataFrame, out_dir: Path) -> None:
    counts = train[TARGET_COL].value_counts().sort_index()
    pct    = train[TARGET_COL].value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame({
        "count": counts,
        "pct":   pct.round(2),
    })
    summary.index.name = "label"
    summary.to_csv(out_dir / "class_balance.csv")
    print(summary.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Non-pond (0)", "Aquaculture pond (1)"],
        counts.values,
        color=["#4c72b0", "#dd8452"],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, count, pct_val in zip(bars, counts.values, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{count}\n({pct_val:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution — Training Set")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'class_balance.png'}")