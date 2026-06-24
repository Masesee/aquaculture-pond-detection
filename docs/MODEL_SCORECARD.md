# Model Scorecard

Single-page reference for the final model's performance, configuration,
and known limitations. Intended for anyone replicating or extending this work.

---

## Identity

| Field | Value |
|---|---|
| Task | Binary classification: aquaculture pond vs other land cover |
| Geography | Two regions, Azerbaijan / Central Asia (~48°E, 39-40°N) |
| Resolution | 10m × 10m per sample |
| Train period | Time period A (exact dates not disclosed by competition) |
| Test period | Time period B (temporal generalization required) |
| Competition | Zindi aquaculture pond detection |

---

## Performance

| Metric | OOF (train period) | Leaderboard (test period) |
|---|---|---|
| Combined score | 0.9908 | **0.9798** |
| F1 @ 0.5 threshold | 0.9847 | **0.9697** |
| ROC-AUC | 0.9986 | **0.9949** |
| Predicted positive rate | 0.408 | 0.409 |
| Predicted ponds | — | 351 / 858 |

OOF overestimates leaderboard by ~0.011 due to temporal distribution shift
between train and test periods.

---

## Feature Summary

**Total features: 203**

| Group | Count | Share (SHAP) | Key features |
|---|---|---|---|
| Water indices (NDWI, MNDWI, AWEInsh) | 36 | 23.4% | NDWI__max, AWEInsh__cv, MNDWI__cv |
| SAR (VH, VV, SAR_diff_db) | 31 | 16.6% | SAR_diff_db__mean, VH__median, VV__median |
| Turbidity + algae (NDTI, re1_nir) | 24 | ~18% | NDTI__min, NDTI__mean_consec_change, re1_nir__max |
| Optical raw bands | 90 | 32.8% | Distributed signal; no single dominant feature |
| Vegetation (NDVI, NDRE) | 19 | 6.8% | NDVI__cv, NDVI__min |
| Spatial | 1 | 1.5% | region (KMeans binary indicator) |
| Temporal stability | Included above | — | consecutive-change stats, monotone fraction |
| Persistence counts | 5 | Included above | NDWI_pos_count, MNDWI_pos_count |

**Aggregation types per band/index:** mean, median, std, min, max, p10, p90, cv, range
**Temporal stability per selected index:** max_consec_change, mean_consec_change, monotone_fraction
**Removed:** dist_to_pond_centroid (caused spatial memorization, rank 1 SHAP before removal)

---

## Model Configuration
Algorithm: LightGBM (gbdt)

| Parameter | Value |
|---|---:|
| Objective | binary |
| n_estimators | 870 |
| learning_rate | 0.10764193150844718 |
| num_leaves | 102 |
| max_depth | 8 |
| min_child_samples | 85 |
| subsample | 0.6474257684746084 |
| colsample_bytree | 0.37320653940385246 |
| reg_alpha | 0.00012975561645961348 |
| reg_lambda | 0.0013218190564466313 |
| class_weight | None |
| random_state | 42 |

**Why these params:** Found by progressive Optuna narrowing.
Global search identified the neighborhood; ±20% local refinement found the optimum.
`colsample_bytree=0.373` is the single most important param — forces feature diversity
across trees, preventing all trees from converging to NDWI-heavy splits.
`min_child_samples=85` prevents overfitting to region-0's 13 pond training examples.

---

## Training Procedure
1. Load raw Sentinel-1/2 tabular data (Train.csv, Test.csv)
2. Assign regions via KMeans(n_clusters=2) on lon/lat
	- Region 0: 321 samples, 4.0% pond rate (northwest)
	- Region 1: 642 samples, 59.2% pond rate (southeast)
3. Compute 203 features via pipelines/features/build_features.py
4. 5-fold CV stratified on label × region interaction strata
5. Train LightGBM with early stopping (50 rounds) per fold
6. Fit isotonic regression calibrator on OOF probabilities
7. Refit final model on full training data using n_estimators=870
8. Apply calibrator to test probabilities
9. Threshold at 0.5 for binary predictions

---

## Known Limitations

**Temporal shift:** OOF-to-leaderboard gap of ~0.011 confirms the model partially
overfits to the training time period despite temporal invariant features.
Features that remain time-sensitive: optical raw band aggregations (32.8% of importance).

**Region 0 weakness:** Fold 0 (highest region-0 validation share) consistently scores
F1=0.9677 vs F1=0.98+ for other folds. With only 13 pond training examples in region-0,
the model has a weak prior for that region.

**OOD test cluster:** ~30-40 test points at lon≈47.6 have no nearby training data.
Predictions for these locations are spectral extrapolation only.

**Spatial prior removed:** dist_to_pond_centroid was deliberately removed to prevent
geographic memorization. The model relies entirely on spectral and temporal features.
If applied to a new geographic region, performance is expected to be lower than the
leaderboard score.

---

## Reproduction

```bash
pip install -r requirements.txt
# Place Train.csv and Test.csv in data/raw/

pytest pipelines/ -v                              # all gate tests must pass
python -m pipelines.features.build_features       # builds 203-feature matrices
python -m pipelines.training.train                # trains final model
# Submission file: outputs/submissions/submission.csv
```

Ensure `outputs/models/best_params.json` contains the sub 22 params above before
running `train.py`. The file is committed to the repository.