"""
SHAP-based feature importance analysis + feature-count sweep.

Loads the saved LightGBM model, deduplicates augmented training rows to one
canonical row per original pond (avoids augmentation bias in SHAP), then:

  1. Computes SHAP values and ranks all features by mean |SHAP|.
  2. Sweeps N = [10,20,30,40,50,60,80,100,120,150,180,all] features using
     GroupKFold (grouped by original pond ID) ? leak-free CV.
  3. Plots the elbow curve to identify the optimal feature count.

Outputs:
  outputs/evaluation/shap_importance.csv       ? ranked mean |SHAP| per feature
  outputs/evaluation/shap_bar_top30.png        ? bar chart, top 30 features
  outputs/evaluation/shap_beeswarm_top20.png   ? beeswarm, top 20 features
  outputs/evaluation/shap_waterfall_pond.png   ? single pond example
  outputs/evaluation/shap_waterfall_nonpond.png ? single non-pond example
  outputs/evaluation/shap_sweep_results.csv    ? CV score vs feature count
  outputs/evaluation/shap_sweep_curve.png      ? elbow plot

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
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL

PROCESSED_DIR  = ROOT / "data"    / "processed"
MODELS_DIR     = ROOT / "outputs" / "models"
EVAL_DIR       = ROOT / "outputs" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _dedup_to_originals(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    The training parquet contains 27x augmented rows (one per window simulation).
    SHAP computed on duplicated rows biases importance toward features that
    discriminate *between windows of the same pond* rather than between ponds.

    Strategy: keep the row whose window best covers the full 12-month span
    (i.e. max window_length, then lowest window_start as tiebreaker).
    This gives us one canonical observation per real pond.
    """
    df = train_df.copy()
    # Extract original pond ID by stripping the _w<N> suffix
    df["_orig_id"] = df["ID"].str.replace(r"_w\d+$", "", regex=True)

    if "window_length" in df.columns:
        df = (
            df.sort_values(["_orig_id", "window_length", "window_start"],
                           ascending=[True, False, True])
              .drop_duplicates(subset="_orig_id", keep="first")
        )
    else:
        df = df.drop_duplicates(subset="_orig_id", keep="first")

    df = df.drop(columns=["_orig_id"])
    return df.reset_index(drop=True)


def main() -> None:
    # -- Load ------------------------------------------------------------------
    print("=== Loading model and features ===")
    model         = joblib.load(MODELS_DIR / "lgbm_model.joblib")
    train_df_full = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")

    # Use the model's own feature list — the parquet may have more columns than
    # the model was trained on (e.g. SWI/NFAI added after invariant_features.txt
    # was frozen). Driving selection from model.feature_name_ keeps them in sync.
    model_feature_cols = list(model.feature_name_)
    print(f"  Model trained on {len(model_feature_cols)} features | "
          f"Parquet has {train_df_full.shape[1] - 2} feature columns")

    # Validate all model features exist in parquet
    missing = [f for f in model_feature_cols if f not in train_df_full.columns]
    if missing:
        raise ValueError(f"Model features missing from parquet: {missing}")

    # -- Deduplicate: one row per original pond (removes augmentation bias) ----
    print("=== Deduplicating augmented rows to one per original pond ===")
    train_df = _dedup_to_originals(train_df_full)
    print(f"  Augmented rows: {len(train_df_full):,} -> Deduplicated: {len(train_df):,} (original ponds)")

    feature_cols = model_feature_cols   # use model's feature set, not all parquet cols
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values

    print(f"  Features used for SHAP: {X.shape[1]} | Ponds: {X.shape[0]}")

    # -- SHAP values -----------------------------------------------------------
    print("=== Computing SHAP values (TreeExplainer on deduplicated data) ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM binary: shap_values may be list [neg_class, pos_class] or single array
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]   # positive class (pond)
    else:
        shap_pos = shap_values

    print(f"  SHAP array shape: {shap_pos.shape}")

    # -- Feature importance table -----------------------------------------------
    mean_abs_shap = np.abs(shap_pos).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    importance_df.to_csv(EVAL_DIR / "shap_importance.csv", index=False)
    print("\n  Top 20 features by mean |SHAP|:")
    print(importance_df.head(20).to_string(index=False))

    # -- Bar chart: top 30 -----------------------------------------------------
    _plot_bar(importance_df.head(30), EVAL_DIR / "shap_bar_top30.png")

    # -- Beeswarm: top 20 ------------------------------------------------------
    _plot_beeswarm(shap_pos, X, feature_cols, importance_df, EVAL_DIR / "shap_beeswarm_top20.png")

    # -- Waterfall: one pond, one non-pond example -----------------------------
    pond_idx    = int(np.where(y == 1)[0][0])
    nonpond_idx = int(np.where(y == 0)[0][0])

    _plot_waterfall(
        explainer, X, pond_idx,
        title="SHAP Waterfall ? Pond Example (label=1)",
        save_path=EVAL_DIR / "shap_waterfall_pond.png",
    )
    _plot_waterfall(
        explainer, X, nonpond_idx,
        title="SHAP Waterfall ? Non-pond Example (label=0)",
        save_path=EVAL_DIR / "shap_waterfall_nonpond.png",
    )

    # -- Feature group summary -------------------------------------------------
    _print_group_summary(importance_df)

    # -- Feature count sweep (elbow analysis) ---------------------------------
    _run_feature_sweep(
        train_df_full=train_df_full,
        importance_df=importance_df,
        feature_cols=feature_cols,
    )

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
    ax.set_title("Feature Importance ? Top 30 (mean |SHAP|)")
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
    ax.set_title("SHAP Beeswarm ? Top 20 Features\n(colour = feature value: blue=low, red=high)")
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
    Groups features by origin and prints total importance share per group.
    Helps identify whether invariant physics features dominate or raw bands do.
    """
    groups = {
        "SAR (VH/VV raw + SAR_diff)": [
            "VH__", "VV__", "SAR_diff", "SAR_diff_neg15"
        ],
        "Water indices (NDWI/MNDWI/AWEInsh/SWI)": [
            "NDWI__", "MNDWI__", "AWEInsh__",
            "NDWI_pos", "MNDWI_pos", "AWEInsh_pos",
            "SWI__", "water_index_agreement", "water_index_unanimous",
        ],
        "Turbidity/Algae (NDTI/re1_nir/NFAI)": [
            "NDTI__", "re1_nir__", "NFAI__",
        ],
        "Vegetation indices (NDVI/NDRE)": [
            "NDVI__", "NDRE__", "NDVI_low",
        ],
        "Optical raw bands": [
            "blue__", "green__", "red__", "re1__",
            "re2__", "re3__", "nir__", "nira__",
            "swir1__", "swir2__",
        ],
        "Window metadata (geography-risk)": [
            "window_start", "window_length", "window_center",
            "window_start_sin", "window_start_cos",
            "window_center_sin", "window_center_cos",
        ],
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
        print(f"  {group_name:<50} {share:5.1f}%  (n={n_feats})")


def _run_feature_sweep(
    train_df_full: pd.DataFrame,
    importance_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """
    Sweeps over N top-SHAP features and measures leak-free GroupKFold CV score.

    GroupKFold groups by original pond ID (stripping _w<N> suffix), so all
    augmented windows of the same real pond always land in the same fold.
    This closes the augmentation leak that makes standard CV overoptimistic.

    Saves:
        outputs/evaluation/shap_sweep_results.csv
        outputs/evaluation/shap_sweep_curve.png
    """
    print("\n=== Running feature count sweep (GroupKFold, leak-free) ===")

    ranked_features = importance_df["feature"].tolist()
    n_total = len(ranked_features)

    # Candidate N values ? always include the full set
    candidates = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, n_total]
    candidates = sorted(set(n for n in candidates if n <= n_total))

    # Groups: original pond ID for every augmented row
    X_full = train_df_full[feature_cols]
    y_full = train_df_full[TARGET_COL].values
    group_ids = train_df_full["ID"].str.replace(r"_w\d+$", "", regex=True).values

    gkf = GroupKFold(n_splits=5)

    # Base LightGBM params ? same as production but lighter for speed
    base_params = dict(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    results = []
    for n in candidates:
        top_n_feats = ranked_features[:n]
        X_n = X_full[top_n_feats]

        oof_probs = np.zeros(len(y_full))
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_n, y_full, groups=group_ids)
        ):
            clf = lgb.LGBMClassifier(**base_params)
            clf.fit(
                X_n.iloc[train_idx], y_full[train_idx],
                eval_set=[(X_n.iloc[val_idx], y_full[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            oof_probs[val_idx] = clf.predict_proba(X_n.iloc[val_idx])[:, 1]

        # Threshold at 0.5 for F1
        oof_preds = (oof_probs >= 0.5).astype(int)
        f1  = f1_score(y_full, oof_preds, zero_division=0)
        auc = roc_auc_score(y_full, oof_probs)
        score = 0.6 * f1 + 0.4 * auc

        results.append({"n_features": n, "f1": f1, "auc": auc, "combined": score})
        print(f"  N={n:4d}  F1={f1:.4f}  AUC={auc:.4f}  Score={score:.4f}")

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(EVAL_DIR / "shap_sweep_results.csv", index=False)
    print("  Saved: shap_sweep_results.csv")

    # -- Elbow plot ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(sweep_df["n_features"], sweep_df["combined"], "o-",
             color="#2563eb", linewidth=2, label="Combined (0.6F1+0.4AUC)")
    ax2.plot(sweep_df["n_features"], sweep_df["auc"], "s--",
             color="#16a34a", linewidth=1.5, alpha=0.7, label="AUC")
    ax2.plot(sweep_df["n_features"], sweep_df["f1"], "^--",
             color="#dc2626", linewidth=1.5, alpha=0.7, label="F1")

    # Mark the elbow: first N where gain over previous is < 0.001
    gains = sweep_df["combined"].diff().fillna(999)
    elbow_mask = gains < 0.001
    if elbow_mask.any():
        elbow_n = int(sweep_df.loc[elbow_mask.idxmax(), "n_features"])
        ax1.axvline(elbow_n, color="orange", linestyle=":", linewidth=2,
                    label=f"Elbow @ N={elbow_n}")

    ax1.set_xlabel("Number of features (top-N by SHAP)")
    ax1.set_ylabel("Combined score (GroupKFold OOF)", color="#2563eb")
    ax2.set_ylabel("AUC / F1", color="grey")
    ax1.set_title("Feature Count Sweep ? GroupKFold (leak-free)\nElbow = point of diminishing returns")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(EVAL_DIR / "shap_sweep_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved: shap_sweep_curve.png")

    # Print elbow summary
    best_row = sweep_df.loc[sweep_df["combined"].idxmax()]
    print(f"\n  Best N: {int(best_row['n_features'])} features  "
          f"Score={best_row['combined']:.4f}  F1={best_row['f1']:.4f}  AUC={best_row['auc']:.4f}")


if __name__ == "__main__":
    main()