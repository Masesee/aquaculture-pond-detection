"""
SHAP-based feature importance analysis.

Loads the saved LightGBM model and OOF predictions, computes SHAP values
on the full training feature matrix, and produces:

  outputs/evaluation/shap_importance.csv     — ranked mean |SHAP| per feature
  outputs/evaluation/shap_bar_top30.png      — bar chart, top 30 features
  outputs/evaluation/shap_beeswarm_top20.png — beeswarm, top 20 features
  outputs/evaluation/shap_waterfall_pond.png — single pond example
  outputs/evaluation/shap_waterfall_nonpond.png — single non-pond example

Run with:
    python -m pipelines.evaluation.shap_analysis
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from contracts.schema import TARGET_COL

PROCESSED_DIR  = ROOT / "data"    / "processed"
MODELS_DIR     = ROOT / "outputs" / "models"
EVAL_DIR       = ROOT / "outputs" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    print("=== Loading model and features ===")
    model    = joblib.load(MODELS_DIR / "lgbm_model.joblib")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")

    feature_cols = [c for c in train_df.columns if c not in ["ID", TARGET_COL]]
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values

    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")

    # ── SHAP values ───────────────────────────────────────────────────────────
    print("=== Computing SHAP values (TreeExplainer) ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM binary: shap_values may be list [neg_class, pos_class] or single array
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]   # positive class (pond)
    else:
        shap_pos = shap_values

    print(f"  SHAP array shape: {shap_pos.shape}")

    # ── Feature importance table ───────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_pos).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    importance_df.to_csv(EVAL_DIR / "shap_importance.csv", index=False)
    print("\n  Top 20 features by mean |SHAP|:")
    print(importance_df.head(20).to_string(index=False))

    # ── Bar chart: top 30 ─────────────────────────────────────────────────────
    _plot_bar(importance_df.head(30), EVAL_DIR / "shap_bar_top30.png")

    # ── Beeswarm: top 20 ──────────────────────────────────────────────────────
    _plot_beeswarm(shap_pos, X, feature_cols, importance_df, EVAL_DIR / "shap_beeswarm_top20.png")

    # ── Waterfall: one pond, one non-pond example ─────────────────────────────
    pond_idx    = np.where(y == 1)[0][0]
    nonpond_idx = np.where(y == 0)[0][0]

    _plot_waterfall(
        explainer, X, pond_idx,
        title="SHAP Waterfall — Pond Example (label=1)",
        save_path=EVAL_DIR / "shap_waterfall_pond.png",
    )
    _plot_waterfall(
        explainer, X, nonpond_idx,
        title="SHAP Waterfall — Non-pond Example (label=0)",
        save_path=EVAL_DIR / "shap_waterfall_nonpond.png",
    )

    # ── Feature group summary ─────────────────────────────────────────────────
    _print_group_summary(importance_df)

    print(f"\n=== SHAP analysis complete. Artefacts in {EVAL_DIR} ===")


def _plot_bar(importance_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(importance_df))
    )
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["mean_abs_shap"][::-1],
        color=colors,
        edgecolor="none",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance — Top 30 (mean |SHAP|)")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_beeswarm(
    shap_pos: np.ndarray,
    X: pd.DataFrame,
    feature_cols: list[str],
    importance_df: pd.DataFrame,
    save_path: Path,
) -> None:
    top20 = importance_df.head(20)["feature"].tolist()
    top20_idx = [feature_cols.index(f) for f in top20]

    shap_top20 = shap_pos[:, top20_idx]
    X_top20    = X[top20].values

    fig, ax = plt.subplots(figsize=(10, 8))
    # Manual beeswarm: scatter with jitter, coloured by feature value
    for i, feat in enumerate(top20[::-1]):  # bottom to top
        feat_idx_orig = top20.index(feat)
        sv   = shap_top20[:, feat_idx_orig]
        fv   = X_top20[:, feat_idx_orig]

        # Normalise feature values to [0,1] for colour mapping
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jitter  = np.random.default_rng(i).uniform(-0.3, 0.3, len(sv))

        sc = ax.scatter(
            sv, np.full_like(sv, i) + jitter,
            c=fv_norm, cmap="coolwarm",
            s=6, alpha=0.5, linewidths=0,
        )

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20[::-1], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Beeswarm — Top 20 Features\n(colour = feature value: blue=low, red=high)")
    plt.colorbar(sc, ax=ax, label="Feature value (normalised)", shrink=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    sample_idx: int,
    title: str,
    save_path: Path,
) -> None:
    explanation = shap.Explanation(
        values          = explainer.shap_values(X.iloc[[sample_idx]]),
        base_values     = explainer.expected_value,
        data            = X.iloc[[sample_idx]].values,
        feature_names   = X.columns.tolist(),
    )
    # Handle list output from binary classifier
    if isinstance(explanation.values, list):
        vals = explanation.values[1][0]
        base = explanation.base_values[1] if hasattr(explanation.base_values, '__len__') else explanation.base_values
    else:
        vals = explanation.values[0]
        base = explanation.base_values if not hasattr(explanation.base_values, '__len__') else explanation.base_values[0]

    exp = shap.Explanation(
        values        = vals,
        base_values   = base,
        data          = X.iloc[sample_idx].values,
        feature_names = X.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _print_group_summary(importance_df: pd.DataFrame) -> None:
    """
    Groups features by origin (raw band, spectral index, SAR, spatial)
    and prints total importance share per group.
    Helps identify whether invariant features dominate or raw bands do.
    """
    groups = {
        "SAR (VH/VV raw + SAR_diff)":    ["VH__", "VV__", "SAR_diff", "SAR_diff_neg15"],
        "Water indices (NDWI/MNDWI/AWEInsh)": ["NDWI__", "MNDWI__", "AWEInsh__",
                                                "NDWI_pos", "MNDWI_pos", "AWEInsh_pos"],
        "Vegetation indices (NDVI/NDRE)": ["NDVI__", "NDRE__", "NDVI_low"],
        "Optical raw bands":              ["blue__", "green__", "red__", "re1__",
                                           "re2__", "re3__", "nir__", "nira__",
                                           "swir1__", "swir2__"],
        "Spatial":                        ["dist_to_pond_centroid", "region"],
    }

    total_importance = importance_df["mean_abs_shap"].sum()
    print("\n  === Feature group importance share ===")
    for group_name, prefixes in groups.items():
        mask = importance_df["feature"].apply(
            lambda f: any(f.startswith(p) or f == p for p in prefixes)
        )
        group_importance = importance_df.loc[mask, "mean_abs_shap"].sum()
        share = 100 * group_importance / total_importance
        n_feats = mask.sum()
        print(f"  {group_name:<45} {share:5.1f}%  (n={n_feats})")


if __name__ == "__main__":
    main()