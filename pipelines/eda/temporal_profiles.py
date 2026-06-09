"""
Monthly temporal profiles: mean band values per class across months.
Shows whether the time series shape differs between ponds and non-ponds
for key bands — critical for confirming temporal invariant assumptions.

Outputs:
  - temporal_profile_{band}.png  for VH, VV, NDWI, NDVI, nir, swir1
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from contracts.schema import MONTHS, TARGET_COL

BANDS_TO_PLOT = ["VH", "VV", "nir", "swir1", "green", "red"]


def _compute_ndwi_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with NDWI_01 ... NDWI_12 columns added."""
    df = df.copy()
    for m in MONTHS:
        df[f"NDWI_{m}"] = (df[f"green_{m}"] - df[f"nir_{m}"]) / (
            df[f"green_{m}"] + df[f"nir_{m}"] + 1e-9
        )
        df[f"NDVI_{m}"] = (df[f"nir_{m}"] - df[f"red_{m}"]) / (
            df[f"nir_{m}"] + df[f"red_{m}"] + 1e-9
        )
    return df


def run_temporal_profiles(train: pd.DataFrame, out_dir: Path) -> None:
    df = _compute_ndwi_monthly(train)
    all_bands = BANDS_TO_PLOT + ["NDWI", "NDVI"]
    month_ints = list(range(1, 13))

    for band in all_bands:
        fig, ax = plt.subplots(figsize=(9, 4))

        for cls, color, label in [(0, "#4c72b0", "Non-pond"), (1, "#dd8452", "Pond")]:
            subset = df[df[TARGET_COL] == cls]
            monthly_cols = [f"{band}_{m}" for m in MONTHS]
            means = subset[monthly_cols].mean().values
            stds  = subset[monthly_cols].std().values

            ax.plot(month_ints, means, color=color, label=label, linewidth=2, marker="o", markersize=4)
            ax.fill_between(
                month_ints,
                means - stds,
                means + stds,
                color=color,
                alpha=0.15,
            )

        ax.set_xlabel("Month")
        ax.set_ylabel(band)
        ax.set_title(f"Monthly Mean ± 1 SD — {band}")
        ax.set_xticks(month_ints)
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / f"temporal_profile_{band}.png", dpi=150)
        plt.close(fig)

    print(f"  Saved: temporal_profile_{{band}}.png for {all_bands}")