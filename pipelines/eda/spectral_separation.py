"""
Q2/Q5: Do pond vs non-pond show distinct NDWI and SAR ratio distributions?
Outputs:
  - ndwi_by_class.png           — violin plot of monthly NDWI per class
  - sar_ratio_by_class.png      — violin plot of monthly SAR ratio per class
  - index_separation_summary.csv — mean ± std per class per index per month
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contracts.schema import MONTHS, TARGET_COL


def _ndwi(df: pd.DataFrame, month: str) -> pd.Series:
    return (df[f"green_{month}"] - df[f"nir_{month}"]) / (
        df[f"green_{month}"] + df[f"nir_{month}"] + 1e-9
    )


def _mndwi(df: pd.DataFrame, month: str) -> pd.Series:
    return (df[f"green_{month}"] - df[f"swir1_{month}"]) / (
        df[f"green_{month}"] + df[f"swir1_{month}"] + 1e-9
    )


def _sar_ratio(df: pd.DataFrame, month: str) -> pd.Series:
    # VH and VV are in dB — convert to linear before ratio
    vh_lin = 10 ** (df[f"VH_{month}"] / 10)
    vv_lin = 10 ** (df[f"VV_{month}"] / 10)
    return vh_lin / (vv_lin + 1e-9)


def run_spectral_separation(train: pd.DataFrame, out_dir: Path) -> None:
    df = train.copy()

    # Build long-form dataframes for violin plots
    records_ndwi  = []
    records_mndwi = []
    records_sar   = []

    for month in MONTHS:
        ndwi  = _ndwi(df, month)
        mndwi = _mndwi(df, month)
        sar   = _sar_ratio(df, month)

        for idx in df.index:
            records_ndwi.append({
                "month": month, "value": ndwi[idx], TARGET_COL: df.loc[idx, TARGET_COL]
            })
            records_mndwi.append({
                "month": month, "value": mndwi[idx], TARGET_COL: df.loc[idx, TARGET_COL]
            })
            records_sar.append({
                "month": month, "value": sar[idx], TARGET_COL: df.loc[idx, TARGET_COL]
            })

    ndwi_df  = pd.DataFrame(records_ndwi)
    mndwi_df = pd.DataFrame(records_mndwi)
    sar_df   = pd.DataFrame(records_sar)

    # --- NDWI violin ---
    _violin_plot(
        ndwi_df, "NDWI", out_dir / "ndwi_by_class.png",
        hline=0.0, hline_label="NDWI=0 (water positive)"
    )

    # --- MNDWI violin ---
    _violin_plot(
        mndwi_df, "MNDWI", out_dir / "mndwi_by_class.png",
        hline=0.0, hline_label="MNDWI=0"
    )

    # --- SAR ratio violin ---
    _violin_plot(
        sar_df, "SAR ratio (VH/VV linear)", out_dir / "sar_ratio_by_class.png",
        hline=None
    )

    # --- Summary CSV ---
    summary_rows = []
    for index_name, long_df in [("NDWI", ndwi_df), ("MNDWI", mndwi_df), ("SAR_ratio", sar_df)]:
        for cls in [0, 1]:
            for month in MONTHS:
                vals = long_df.loc[
                    (long_df[TARGET_COL] == cls) & (long_df["month"] == month),
                    "value"
                ]
                summary_rows.append({
                    "index": index_name,
                    "class": cls,
                    "month": month,
                    "mean":  vals.mean(),
                    "std":   vals.std(),
                    "median": vals.median(),
                })

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "index_separation_summary.csv", index=False
    )
    print("  Saved: ndwi_by_class.png, mndwi_by_class.png, sar_ratio_by_class.png")
    print("  Saved: index_separation_summary.csv")


def _violin_plot(
    long_df: pd.DataFrame,
    index_name: str,
    save_path: Path,
    hline: float | None,
    hline_label: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    class_labels = {0: "Non-pond", 1: "Aquaculture Pond"}
    colors = {0: "#4c72b0", 1: "#dd8452"}

    for ax, cls in zip(axes, [0, 1]):
        subset = long_df[long_df[TARGET_COL] == cls]
        sns.violinplot(
            data=subset, x="month", y="value",
            color=colors[cls], ax=ax,
            inner="quartile", linewidth=0.8,
        )
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--", linewidth=1, label=hline_label)
            ax.legend(fontsize=7)
        ax.set_title(f"{index_name} — {class_labels[cls]}")
        ax.set_xlabel("Month")
        ax.set_ylabel(index_name if cls == 0 else "")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"{index_name} Distribution by Month and Class", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)