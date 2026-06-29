# Revised Challenge Context & Modeling Strategy

This document details the updates made to the Aquaculture Pond Detection project following the organizers' revised challenge setup.

---

## 1. The Revised Challenge Setup

### Data Changes
* **No Coordinates**: `latitude` and `longitude` metadata columns have been completely removed.
* **Dataset Rescale**:
  * **Train Set**: Expanded to **1,821 samples** (contains the original training set + the original test set with labels).
  * **Test Set**: **1,030 samples** (completely new data).
* **Temporal Masking**: Test samples are masked to simulate a real-world partial-observation setting. Each test sample only includes a consecutive block of **4, 5, or 6 months** of observations; the remaining months are set to `-9999` across all 12 bands.

---

## 2. Our Modeling Approach

To address coordinate removal and temporal missingness, we implemented a robust pipeline structured as follows:

```mermaid
graph TD
    A[Raw Train: 1,821 rows] --> B[Temporal Mask Augmentation: 49,167 rows]
    B --> C[Vectorized NaN-Safe Feature Building: 214 features]
    C --> D[Kolmogorov-Smirnov Drift Audit & Pruning: KS < 0.20]
    D --> E[Pruned Domain-Invariant Feature Subset: 146 features]
    E --> F[Single-Window Stratified Group CV]
    F --> G1[LightGBM Classifier]
    F --> G2[XGBoost Classifier]
    F --> G3[CatBoost Classifier]
    G1 --> H[1/3 Triad Blended Ensemble & Prior Shift Correction]
    G2 --> H
    G3 --> H
    H --> I[Final Calibrated Submission]
```

### 2.1 NaN-Safe Feature Engineering & Trend Extraction
* All raw `-9999` masks are replaced with `NaN` at load time.
* Temporal aggregations (means, medians, standard deviations, percentiles) are calculated row-wise ignoring `NaN` values.
* **Vectorized Linear Trend Slopes (`__trend_slope`)**: Added linear slope extraction ($\Delta y / \Delta t$) across valid unmasked months for key indices (`NDWI`, `MNDWI`, `NDTI`, `re1_nir`, `SAR_diff_db`) to capture dynamic water filling/harvesting trajectories.
* All occurrence counts (like months meeting water thresholds) are normalized as fractions of observed months.

### 2.2 Temporal Mask Augmentation
* To prevent distribution shifts on statistical aggregations between 12-month training profiles and 4–6 month test profiles, the 1,821 training profiles were expanded to **49,167 samples** by simulating all 24 possible consecutive 4, 5, or 6-month observation windows.

### 2.3 Single-Window Stratified Group CV
* To prevent target leakage across augmented duplicates of the same training sample while mirroring test-time evaluation, CV splits are formed on a 1-window-per-sample subset (1,821 validation rows) grouped by original sample ID.

### 2.4 Empirical KS Drift Pruning
* Applied Kolmogorov-Smirnov (KS) two-sample testing between training and test distributions.
* Features exhibiting severe distribution drift ($\text{KS} \ge 0.20$, such as raw shifted radar backscatter levels `SAR_diff_db__mean` and raw vegetation ranges) are systematically pruned.
* Leaving **146 strictly domain-invariant features** in `invariant_features.txt`.

### 2.5 Multi-Model Triad Ensemble (LightGBM + XGBoost + CatBoost)
* **LightGBM:** Leaf-wise histogram splits with `colsample_bytree = 0.5205` (~76 features observed per split).
* **XGBoost:** Depth-wise greedy tree growth (`train_xgb.py`, version 3.2.0) trained on the 146 invariant features.
* **CatBoost:** Symmetric oblivious trees (`train_catboost.py`, version 1.2.10) offering structural orthogonality.
* **Triad Blending & Clean Contract (`blend_ensemble.py`):** Equal 1/3 probability blending across all three model families on pure calibrated test probabilities, applying prior shift correction exactly once on the blended matrix.

---

## 3. Submission History (Revised Phase)

| Submission | Configuration | OOF Score | LB Score | LB AUC | LB F1 | Predicted Ponds |
|---|---|---|---|---|---|---|
| Phase 2 Baseline | All 209 features, high capacity parameters | 0.9791 | 0.8303 | 0.8432 | 0.8217 | 634 / 1030 |
| Phase 2 Invariant | 84 robust features (pruned), high capacity | 0.9715 | 0.8316 | 0.8277 | 0.8342 | 637 / 1030 |
| Quantile Warping Bug | Independent ECDF transform on train/test | 0.9740 | 0.7981 | 0.8102 | 0.7902 | 643 / 1030 |
| KS Pruned + Scale Fix | 83 invariant features, no quantile, colsample=0.90 | 0.9739 | 0.8412 | 0.8459 | 0.8380 | 643 / 1030 |
| Trend Slopes (Sub 35) | 146 invariant features + trend slopes, LGBM single | 0.9812 | 0.8539 | 0.8719 | 0.8418 | 653 / 1030 |
| 2-Way Ensemble (Sub 36) | 50/50 LightGBM + XGBoost Ensemble on 146 features | 0.9813 | 0.8631 | 0.8817 | 0.8507 | 678 / 1030 |
| 3-Way Triad Bug (Sub 37) | Mixed prior correction contract on test | 0.9813 | 0.8626 | **0.8846** | 0.8479 | 678 / 1030 |
| **Clean Triad (Sub 38)** | **Clean 1/3 Triad Ensemble (LGBM + XGB + CB) on 146 feat** | **0.9813** | *TBD* | *TBD* | *TBD* | **667 / 1030** |

---

## 4. Next Steps for Top Ranks (Target: LB 0.924+)

1. **Seasonal Z-Score Pre-normalization:** Standardize monthly bands relative to annual population monthly means before computing window aggregations.
