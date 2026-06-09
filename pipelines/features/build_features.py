"""
Feature pipeline entrypoint.

Fits the region model on train coordinates, applies it to both splits,
builds feature matrices, and saves to data/processed/.

Run with:
    python -m pipelines.features.build_features

Outputs:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    outputs/features/feature_pipeline_meta.json   ← centroid, region model params
    outputs/features/feature_names.txt            ← ordered feature list
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from contracts.schema import DataSchema, TARGET_COL
from pipelines.features.aggregations import build_feature_matrix, feature_names, POND_CLUSTER_CENTROID

TRAIN_PATH     = ROOT / "data" / "raw" / "Train.csv"
TEST_PATH      = ROOT / "data" / "raw" / "Test.csv"
PROCESSED_DIR  = ROOT / "data" / "processed"
FEATURES_DIR   = ROOT / "outputs" / "features"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=== Loading raw data ===")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    DataSchema.validate_train(train)
    DataSchema.validate_test(test)
    print(f"Train: {train.shape} | Test: {test.shape}")

    # ── Fit region model on train, apply to both ──
    # KMeans is fit only on train to avoid leakage.
    # Test regions are assigned by nearest centroid from the train-fit model.
    print("\n=== Assigning regions ===")
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(train[["lon", "lat"]].values)

    train_regions = pd.Series(km.predict(train[["lon", "lat"]].values), index=train.index)
    test_regions  = pd.Series(km.predict(test[["lon",  "lat"]].values), index=test.index)

    # Ensure region 1 = high-pond-rate region (59% ponds).
    # If KMeans assigns arbitrarily, we remap so region 1 always means "high pond density".
    region0_pond_rate = train.loc[train_regions == 0, TARGET_COL].mean()
    region1_pond_rate = train.loc[train_regions == 1, TARGET_COL].mean()

    if region0_pond_rate > region1_pond_rate:
        # Swap labels so region 1 = high pond rate, always
        train_regions = 1 - train_regions
        test_regions  = 1 - test_regions
        print("  Region labels swapped: region 1 now = high-pond-rate region")

    for r in [0, 1]:
        mask = train_regions == r
        rate = train.loc[mask, TARGET_COL].mean()
        print(f"  Train Region {r}: n={mask.sum()} | pond rate={rate:.1%}")

    # ── Build feature matrices ──
    print("\n=== Building feature matrices ===")
    train_feats = build_feature_matrix(train, train_regions)
    test_feats  = build_feature_matrix(test,  test_regions)

    # Attach label to train features
    train_feats[TARGET_COL] = train[TARGET_COL].values

    print(f"  Train features shape: {train_feats.shape}")
    print(f"  Test  features shape: {test_feats.shape}")

    # Validate feature count matches contract
    expected = feature_names(exclude_id=True)
    actual   = [c for c in train_feats.columns if c not in ["ID", TARGET_COL]]
    assert actual == expected, (
        f"Feature column mismatch.\n"
        f"Expected {len(expected)}, got {len(actual)}\n"
        f"Missing: {set(expected) - set(actual)}\n"
        f"Extra:   {set(actual) - set(expected)}"
    )
    print(f"  Feature contract validated: {len(expected)} features")

    # ── Save ──
    print("\n=== Saving ===")
    train_feats.to_parquet(PROCESSED_DIR / "train_features.parquet", index=False)
    test_feats.to_parquet(PROCESSED_DIR  / "test_features.parquet",  index=False)
    print("  Saved: data/processed/train_features.parquet")
    print("  Saved: data/processed/test_features.parquet")

    # Save KMeans model for inference reproducibility
    joblib.dump(km, FEATURES_DIR / "region_kmeans.joblib")

    # Save metadata
    meta = {
        "pond_cluster_centroid": list(POND_CLUSTER_CENTROID),
        "n_features": len(expected),
        "feature_names": expected,
        "region_kmeans_centroids": km.cluster_centers_.tolist(),
        "region_pond_rates": {
            "0": float(train.loc[train_regions == 0, TARGET_COL].mean()),
            "1": float(train.loc[train_regions == 1, TARGET_COL].mean()),
        },
    }
    with open(FEATURES_DIR / "feature_pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save ordered feature names as plain text for easy inspection
    with open(FEATURES_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(expected))

    print("  Saved: outputs/features/feature_pipeline_meta.json")
    print("  Saved: outputs/features/feature_names.txt")
    print("\n=== Feature pipeline complete ===")


if __name__ == "__main__":
    main()