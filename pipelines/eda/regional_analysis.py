"""
Q4: Do coordinates reveal two distinct regions?
    Do those regions have different class distributions?
Outputs:
  - region_map.png          — scatter of lat/lon coloured by class + region
  - region_class_balance.csv
  - region_assignments_train.csv  — ID → region label (used downstream)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from contracts.schema import TARGET_COL


N_REGIONS = 2  # problem statement says two pilot regions


def assign_regions(df: pd.DataFrame) -> pd.Series:
    """
    Fit KMeans on lon/lat to assign region labels.
    Deterministic via fixed random_state.
    Returns integer Series aligned with df.index.
    """
    coords = df[["lon", "lat"]].values
    km = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10)
    return pd.Series(km.fit_predict(coords), index=df.index, name="region")


def run_regional_analysis(
    train: pd.DataFrame, test: pd.DataFrame, out_dir: Path
) -> None:
    train = train.copy()
    test  = test.copy()

    train["region"] = assign_regions(train)
    test["region"]  = assign_regions(test)

    # Print region summaries
    for region_id in sorted(train["region"].unique()):
        subset = train[train["region"] == region_id]
        pond_rate = subset[TARGET_COL].mean() * 100
        print(
            f"  Region {region_id}: n={len(subset)} | "
            f"pond rate={pond_rate:.1f}% | "
            f"lat=[{subset['lat'].min():.3f}, {subset['lat'].max():.3f}] "
            f"lon=[{subset['lon'].min():.3f}, {subset['lon'].max():.3f}]"
        )

    # Class balance per region
    balance = (
        train.groupby("region")[TARGET_COL]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
    )
    balance.to_csv(out_dir / "region_class_balance.csv", index=False)

    # Save region assignments for use in feature pipeline
    train[["ID", "region"]].to_csv(
        out_dir / "region_assignments_train.csv", index=False
    )
    test[["ID", "region"]].to_csv(
        out_dir / "region_assignments_test.csv", index=False
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {0: "#4c72b0", 1: "#dd8452"}
    region_shapes = {0: "o", 1: "s"}

    for cls in [0, 1]:
        for reg in sorted(train["region"].unique()):
            mask = (train[TARGET_COL] == cls) & (train["region"] == reg)
            label = f"{'Pond' if cls == 1 else 'Non-pond'}, Region {reg}"
            ax.scatter(
                train.loc[mask, "lon"],
                train.loc[mask, "lat"],
                c=colors[cls],
                marker=region_shapes[reg],
                alpha=0.6,
                s=18,
                label=label,
            )

    # Overlay test points
    ax.scatter(
        test["lon"], test["lat"],
        c="grey", marker="x", alpha=0.3, s=12, label="Test (unlabelled)"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution: Train + Test by Class and Region")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "region_map.png", dpi=150)
    plt.close(fig)
    print("  Saved: region_map.png")


def run_ood_check(
    train: pd.DataFrame, test: pd.DataFrame, out_dir: Path
) -> None:
    """
    Flags test points that fall outside the convex hull of training coordinates.
    Specifically investigates the isolated cluster at lon≈47.6.

    Outputs:
      - ood_test_points.csv        — test IDs flagged as out-of-distribution
      - ood_spatial_check.png      — scatter highlighting OOD points
    """
    from scipy.spatial import ConvexHull

    train_coords = train[["lon", "lat"]].values
    test_coords  = test[["lon",  "lat"]].values

    hull = ConvexHull(train_coords)

    def _in_hull(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
        """Returns boolean array: True if point is inside the convex hull."""
        from scipy.spatial import Delaunay
        tri = Delaunay(train_coords[hull.vertices])
        return tri.find_simplex(points) >= 0

    in_hull_mask = _in_hull(test_coords, hull)
    ood_mask     = ~in_hull_mask

    ood_df = test[ood_mask][["ID", "lon", "lat"]].copy()
    ood_df["ood_flag"] = True
    ood_df.to_csv(out_dir / "ood_test_points.csv", index=False)

    n_ood = ood_mask.sum()
    n_total = len(test)
    print(f"  OOD test points: {n_ood} / {n_total} ({100*n_ood/n_total:.1f}%)")
    if n_ood > 0:
        print(f"  OOD lon range: [{ood_df['lon'].min():.3f}, {ood_df['lon'].max():.3f}]")
        print(f"  OOD lat range: [{ood_df['lat'].min():.3f}, {ood_df['lat'].max():.3f}]")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw convex hull boundary
    hull_pts = train_coords[hull.vertices]
    hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)  # close the polygon
    ax.plot(hull_pts[:, 0], hull_pts[:, 1], "k--", linewidth=1, alpha=0.5, label="Train convex hull")

    # Train points
    ax.scatter(
        train["lon"], train["lat"],
        c="#4c72b0", s=12, alpha=0.4, label="Train"
    )

    # In-distribution test points
    ax.scatter(
        test.loc[in_hull_mask, "lon"], test.loc[in_hull_mask, "lat"],
        c="grey", marker="x", s=14, alpha=0.4, label="Test (in-distribution)"
    )

    # OOD test points — highlighted
    if n_ood > 0:
        ax.scatter(
            ood_df["lon"], ood_df["lat"],
            c="red", marker="*", s=60, zorder=5, label=f"Test OOD (n={n_ood})"
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("OOD Check: Test Points vs Training Convex Hull")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "ood_spatial_check.png", dpi=150)
    plt.close(fig)
    print("  Saved: ood_spatial_check.png")