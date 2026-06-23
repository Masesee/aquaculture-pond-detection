# GeoAI Aquaculture Pond Identification — Full Project Context

This document is a complete, self-contained briefing of the aquaculture pond detection project.
It is intended as starting context for any agent picking up this work. Nothing is assumed. Nothing is omitted.

---

## 1. Competition Context

**Platform:** Zindi
**Task:** Binary classification — given a tabular row representing a 10 m × 10 m geographic point,
predict whether it is an aquaculture pond (label=1) or not (label=0).

**Evaluation metric:** `0.6 × F1 + 0.4 × AUC-ROC` computed at a fixed 0.5 decision threshold.
The submission file must contain two columns: `TargetF1` (binary prediction) and `TargetRAUC` (probability).

**Leaderboard reference:** The top public leaderboard score is 1.000, which is almost certainly
test set memorization. It is not a realistic target. The meaningful competition is in the 0.95–0.98 range.

**Geography:** Two spatial clusters in the Azerbaijan / Central Asia region, approximately 48°E, 39–40°N.
One cluster is in the northwest (low pond density), one in the southeast (high pond density).

---

## 2. Dataset

**Files:**
- `data/raw/Train.csv` — 963 rows, not committed to git
- `data/raw/Test.csv` — 858 rows, not committed to git

**Columns in both files:**
- `ID` — unique identifier per sample (string)
- `lon`, `lat` — geographic coordinates (float)
- 144 raw band columns: `{band}_{month}` for each of 12 bands × 12 months

**Additional column in Train.csv only:**
- `label` — binary target (0 = not a pond, 1 = pond)

**Band names (12 total):**

SAR bands (Sentinel-1, values in dB, negative floats):
- `VH` — vertical transmit, horizontal receive backscatter
- `VV` — vertical transmit, vertical receive backscatter

Optical bands (Sentinel-2 L2A surface reflectance, integer units scaled ×10000):
- `blue`  — B2  (~490 nm)
- `green` — B3  (~560 nm)
- `red`   — B4  (~665 nm)
- `re1`   — B5  (~705 nm, red edge 1)
- `re2`   — B6  (~740 nm, red edge 2)
- `re3`   — B7  (~783 nm, red edge 3)
- `nir`   — B8  (~842 nm, broad NIR)
- `nira`  — B8A (~865 nm, narrow NIR)
- `swir1` — B11 (~1610 nm)
- `swir2` — B12 (~2190 nm)

**Month convention:** months are zero-padded integers from `"01"` to `"12"`,
so a column looks like `green_03` or `VH_11`.

**Class balance:**
- Training set: 963 samples total
- Positive (pond) rate: 40.8% — near-balanced, NOT a severe imbalance problem
- This near-balance is consequential: `class_weight="balanced"` was shown to HURT performance
  because it overweights ponds by 1.22x, pushing borderline probabilities above 0.5

**Temporal split:** Train and test cover different time periods (A and B). The exact dates are not
disclosed by the competition. This temporal distribution shift is the main generalization challenge.
OOF scores consistently overestimate leaderboard performance by ~0.011–0.05 points.

---

## 3. Regional Structure

KMeans(n_clusters=2, random_state=42, n_init=10) is fit on `[lon, lat]` from training data only
(no leakage to test). Test regions are assigned by nearest centroid.

A label-swap ensures region 1 always means high-pond-density:

- **Region 0 (NW):** 321 training samples, 4.0% pond rate
- **Region 1 (SE):** 642 training samples, 59.2% pond rate

This regional split is a critical structural feature of the problem. The model must not mistake
spatial location for causal signal. The KMeans region indicator (binary 0/1) is kept as a feature.
The log-transformed distance to pond centroid (`dist_to_pond_centroid`) was originally included but
was removed in feature version v5 because it was the top SHAP feature by a wide margin — it was
memorizing training pond locations, not learning any physics. When removed, OOF dropped ~0.001 but
leaderboard score was unchanged, confirming the feature was adding spatial memorization not signal.

---

## 4. Feature Engineering Pipeline

**Entry point:** `python -m pipelines.features.build_features`
**Source files:**
- `pipelines/features/build_features.py` — orchestration
- `pipelines/features/indices.py` — spectral index functions
- `pipelines/features/aggregations.py` — temporal aggregation logic
- `contracts/schema.py` — column name contract

### 4.1 Spectral Indices Computed

All indices are computed per month (producing 12 monthly values per index per sample).
Index functions are pure (no side effects, no state).

| Index | Formula | Physical meaning |
|---|---|---|
| NDWI | (green - nir) / (green + nir) | Water detector. Positive = water. |
| MNDWI | (green - swir1) / (green + swir1) | Modified water index; better suppression of built-up false positives. |
| NDVI | (nir - red) / (nir + red) | Vegetation. Near-zero or negative for water. High for dense canopy. |
| NDRE | (nira - re1) / (nira + re1) | Red edge. Low and stable for water; rises for active vegetation. |
| AWEInsh | 4*(green-swir1) - (0.25*nir + 2.75*swir2) / 10000 | Automated water extraction (no shadow variant). Strongly positive for open water. |
| SAR_diff_db | VH - VV | Very negative for specular water surfaces. Less negative for rough terrain/vegetation. |
| NDTI | (red - green) / (red + green) | Turbidity index. POSITIVE for turbid aquaculture water (biological load). NEGATIVE for clear water bodies. |
| re1_nir | re1 / nir | Red edge to NIR ratio. Elevated for algae/phytoplankton in aquaculture ponds. Near 1.0 signals chlorophyll fluorescence. |
| NDWI2 | (nir - swir1) / (nir + swir1) | NIR-SWIR1 water index (Gao 1996). Strongest new addition in v6.1 — SHAP rank #2 (NDWI2__std), total SHAP=2.725. |
| SAR_RVI | 4*VH_lin / (VH_lin + VV_lin) | Linear-power SAR vegetation index. Differentiates flooded vegetation from open water. SHAP rank #6 (SAR_RVI__mean), total SHAP=1.697. |
| SABI | (nir - red) / (blue + green) | Surface Algal Bloom Index. Sensitive to algal blooms in productive water. SHAP rank #17, total SHAP=1.146. |
| CI | re3 / re2 - 1 | Red-Edge Chlorophyll Index. Captures chlorophyll-a fluorescence signal. SHAP total=0.736, rank ~61+. |

All divide-by-zero cases are protected by `EPS = 1e-9`.

**Removed index (v6.1):** `CDOM` (`blue/red`) was trialled in v6 but removed in v6.1 after SHAP analysis confirmed it was noise (total SHAP=0.574, max rank #70). Do not re-add.

**NDTI was the single most impactful feature addition in the entire project** (Sub 12: +0.013 combined
score, +0.009 F1 from adding NDTI alone). The physical reason: fish waste and algae in shallow
productive water raise red reflectance relative to green, producing a uniquely positive NDTI signal.
Clear water bodies (rivers, reservoirs) have negative NDTI. This separates aquaculture from all
other water surfaces.

### 4.2 Temporal Aggregations

For every raw band and every computed index, 9 scalar statistics are computed across the 12 monthly values:

1. `mean`
2. `median`
3. `std` (sample std, ddof=1)
4. `min`
5. `max`
6. `p10` (10th percentile)
7. `p90` (90th percentile)
8. `cv` (coefficient of variation = std / |mean|; abs(mean) handles negative dB values)
9. `range` (max - min)

Feature name convention: `{band_or_index}__{agg_suffix}`, e.g. `NDWI__max`, `VV__cv`, `NDTI__mean`.

### 4.3 Temporal Stability Features

For 5 selected bands/indices, 3 additional statistics over 11 consecutive-month absolute differences
(|value[t+1] - value[t]| for t in 1..11):

Targets: `NDWI`, `MNDWI`, `VV`, `NDTI`, `re1_nir`, `SABI`, `CI` *(SABI and CI added in v6.1)*

Per target:
- `{target}__max_consec_change` — worst single-month jump
- `{target}__mean_consec_change` — average month-to-month variation
- `{target}__monotone_fraction` — fraction of consecutive differences < 0.05 (stability proxy)

### 4.4 Persistence Count Features

Binary threshold counts over 12 months:

| Feature | Definition |
|---|---|
| `NDWI_pos_count` | Months where NDWI > 0 |
| `MNDWI_pos_count` | Months where MNDWI > 0 |
| `NDVI_low_count` | Months where NDVI < 0.1 |
| `AWEInsh_pos_count` | Months where AWEInsh > 0 |
| `SAR_diff_neg15_count` | Months where SAR_diff_db < -15 (very strong water SAR signal) |
| `SABI_pos_count` | Months where SABI > 0 *(added v6.1)* |
| `SAR_RVI_low_count` | Months where SAR_RVI < threshold indicating low vegetation fraction *(added v6.1)* |

### 4.5 Cross-Index Water Agreement Features

- `water_index_agreement` — fraction of months where ALL THREE water indices (NDWI, MNDWI, AWEInsh) are simultaneously positive
- `water_index_unanimous` — fraction of months where all three are simultaneously either all positive or all negative

### 4.6 Spatial Feature

- `region` — KMeans binary indicator (0 or 1). The ONLY spatial feature in the final model.

### 4.7 Removed Feature (Important Historical Note)

`dist_to_pond_centroid` — log1p-transformed Euclidean distance from (lon, lat) to the manually
estimated pond cluster centroid at (48.85, 39.48).

This was REMOVED in v5. It was the #1 SHAP feature before removal (SHAP value 1.548), but it was
memorizing training pond locations. After removal, OOF dropped 0.0012 and leaderboard was identical.
The code is commented out in `aggregations.py` — do not re-enable unless you have a principled
reason.

### 4.8 Feature Version History

| Version | Count | Key additions |
|---|---|---|
| v1 | 144 | 12 bands × 12 months (raw only) |
| v2 | 169 | + temporal aggregations (8 stats) + indices (NDWI, MNDWI, NDVI, NDRE, AWEInsh, SAR_diff_db) + 3 persistence counts + region + dist_to_pond_centroid |
| v3 | 180 | + consecutive-change features (NDWI, MNDWI, VV) + cross-index water agreement |
| v4 | 204 | + NDTI + re1_nir + their 9 aggregations + 3 stability features each |
| v5 (final) | 203 | v4 minus dist_to_pond_centroid |
| v6.1 | 247 | v5 + NDWI2 + SAR_RVI + SABI + CI (all × 9 aggs) + stability for SABI/CI + 2 persistence counts - CDOM |
| v7 | 295 | v6.1 + quarterly aggregations for 6 indices (NDWI, MNDWI, NDTI, re1_nir, SAR_RVI, SABI): Q1–Q4 mean and max (48 features). Sub 27 confirmed this HURT leaderboard: OOF-to-LB gap 0.031 (3× normal). Do not use. |

### 4.9 Optional SHAP Filter

If `outputs/evaluation/shap_importance.csv` exists (from a prior SHAP analysis run),
`build_features.py` automatically applies a top-80 filter. The filter reduces the feature matrix
to the 80 highest mean |SHAP| features, plus any `always_keep` columns.

**CRITICAL RULE:** The SHAP filter must always be computed on the same model it will be applied to.
Applying a SHAP filter from model A to model B caused regression twice in this project
(Submissions 2 and 20). Never reuse stale SHAP rankings across model changes.

### 4.10 Output Files

- `data/processed/train_features.parquet`
- `data/processed/test_features.parquet`
- `outputs/features/feature_pipeline_meta.json` — centroid, region model params, is_shap_filtered flag
- `outputs/features/feature_names.txt` — ordered feature list
- `outputs/features/region_kmeans.joblib` — fitted KMeans for reproducibility

---

## 5. Cross-Validation Strategy

**Entry point:** `pipelines/training/cv_strategy.py`

5-fold stratified CV. The stratification key is a **label × region interaction stratum**:
- Stratum 0: label=0, region=0
- Stratum 1: label=0, region=1
- Stratum 2: label=1, region=0
- Stratum 3: label=1, region=1

This guarantees each fold sees both regions and both classes in proportional representation.
Region 0 has only 4% ponds (13 pond samples out of 321). Standard label-only stratification
would unevenly distribute these rare region-0 ponds across folds. The interaction stratum prevents this.

---

## 6. Primary Model — LightGBM

**Algorithm:** LightGBM gradient boosted decision trees, binary objective, gbdt boosting type.

### 6.1 Training Procedure (train.py)

1. Load `data/processed/train_features.parquet` and `test_features.parquet`
2. Load `outputs/models/best_params.json` if it exists; merge with defaults (tuned params override)
3. Generate 5-fold CV splits on label × region strata
4. OOF loop: for each fold, train with n_estimators=1000 + early stopping (50 rounds patience);
   collect OOF probabilities on validation fold
5. Compute aggregate OOF F1, AUC, combined score
6. Fit isotonic regression calibrator on the 963 OOF probabilities
7. Apply calibrator, compute calibrated OOF metrics
8. Final refit: train on full training data. n_estimators is taken from `best_params.json` if
   present (i.e., the Optuna value), NOT from the mean of early-stopping iterations. This was a
   critical bug fix — prior code used the early-stopping mean, which undertrained the final model
   (confirmed by Submission 16 vs 18 comparison: bug caused -0.006 combined score).
9. Generate raw test probabilities, apply calibrator, threshold at 0.5 for binary predictions
10. Write `outputs/submissions/submission.csv` with columns: `ID`, `TargetF1`, `TargetRAUC`

### 6.2 Default Hyperparameters (fallback if no Optuna output)

```
n_estimators: 1000 (high; early stopping controls actual count)
learning_rate: 0.02
num_leaves: 31
max_depth: -1 (unlimited)
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 1.0
class_weight: "balanced"
random_state: 42
```

Note: the defaults use `class_weight="balanced"`. When Optuna params are loaded, `train.py`
hardcodes `class_weight=None` (line 292 in train.py). This was an intentional fix after Sub 4
showed that balanced weighting hurts F1 on near-balanced data.

### 6.3 Final Model Hyperparameters (Sub 22, Best Submission)

```
n_estimators: 870
learning_rate: 0.10764193150844718
num_leaves: 102
max_depth: 8
min_child_samples: 85
subsample: 0.6474257684746084
colsample_bytree: 0.37320653940385246
reg_alpha: 0.00012975561645961348
reg_lambda: 0.0013218190564466313
class_weight: None
random_state: 42
CV: 5-fold stratified on label × region
Calibration: Isotonic regression on OOF
```

**Why colsample_bytree=0.373 matters:** With 203 features, the default colsample (~1.0) means all
trees see all features, which causes every tree to pick NDWI-heavy splits (NDWI is the strongest
individual predictor). At colsample=0.373, each tree sees only ~76 randomly sampled features,
creating an internal ensemble of diverse feature-subset models. This is the single most important
hyperparameter for generalization in this problem.

**Why min_child_samples=85 matters:** Region 0 has only 13 pond training examples. Without a high
min_child_samples, the model overfits specifically to these 13 region-0 pond locations.

### 6.4 Final Model Performance

| Metric | OOF (train period) | Leaderboard (test period) |
|---|---|---|
| Combined score | 0.9908 | 0.9798 |
| F1 @ 0.5 | 0.9847 | 0.9697 |
| ROC-AUC | 0.9986 | 0.9949 |
| Predicted pond rate | 0.408 | 0.409 |
| Predicted ponds | — | 351 / 858 |

OOF overestimates leaderboard by ~0.011 (temporal distribution shift between train and test periods).

---

## 7. Probability Calibration

**Entry point:** `pipelines/training/calibration.py`
**Method:** Isotonic regression (`sklearn.isotonic.IsotonicRegression`, `out_of_bounds="clip"`)

Calibrator is fit on the 963 OOF probabilities against true labels. Applied to test probabilities.

**Why isotonic over Platt scaling (sigmoid):**
- LightGBM miscalibration is rarely sigmoid-shaped
- Isotonic is non-parametric — no shape assumption
- 963 OOF samples is enough data for isotonic to fit without overfitting

**Why calibrate at all given near-balanced classes:**
- F1 at 0.5 threshold is 60% of the score — a small shift in the probability distribution
  changes ~50 predictions, which can move F1 by 0.004–0.010
- Zero downside if already well-calibrated

**Outputs:**
- `outputs/models/calibrator.joblib`
- `outputs/models/calibration_summary.csv` — bin-level predicted vs actual positive rate

---

## 8. Hyperparameter Tuning — Optuna

**Entry point:** `python -m pipelines.training.tune`
**Source:** `pipelines/training/tune.py`

### 8.1 Current Search Space (Post-Sub-22, Narrow Refinement)

The current `tune.py` implements a narrow ±20% search centered on the Sub 22 best parameters.
This replaced the original global search that proved less effective.

```
n_estimators:      [750, 980]       (Sub 22 value: 870)
learning_rate:     [0.085, 0.130]   (Sub 22 value: 0.1076, log scale)
num_leaves:        [85, 120]        (Sub 22 value: 102)
max_depth:         [6, 10]          (Sub 22 value: 8)
min_child_samples: [70, 100]        (Sub 22 value: 85)
subsample:         [0.55, 0.75]     (Sub 22 value: 0.647)
colsample_bytree:  [0.32, 0.43]     (Sub 22 value: 0.373)
reg_alpha:         [5e-5, 5e-4]     (Sub 22 value: 1.3e-4, log scale)
reg_lambda:        [4e-4, 5e-3]     (Sub 22 value: 1.3e-3, log scale)
class_weight:      "balanced"       (fixed in tune.py; overridden to None in train.py)
```

**N_TRIALS:** 200 (increased from 150)
**Objective:** maximize `0.6 × F1 + 0.4 × AUC` on OOF predictions
**Storage:** SQLite at `experiments/logs/optuna_study.db` (resumable)
**Output:** `outputs/models/best_params.json`

Note: The tune.py objective uses `class_weight="balanced"` but train.py overrides it to `None`.
This is a minor inconsistency — the tuning objective does not perfectly match the training objective.
It has not been fixed because Sub 22 was produced under this configuration.

### 8.2 Optuna Evolution

- **Global search v1** (Sub 2–3): n_estimators [500,3000], lr [0.01,0.15], leaves [31,255],
  depth [3,15], min_child [20,120], subsample [0.5,1.0], colsample [0.3,1.0],
  reg_alpha [1e-4,10.0], reg_lambda [1e-4,10.0]. 150 trials.
  Result: Sub 14 = 0.9720.

- **Global search v3** (Sub 16, 18): Wider bounds. N_TRIALS expanded. Fixed a bug where
  n_estimators tuned by Optuna was ignored in train.py (final fit used early-stopping mean instead).
  After bug fix: Sub 18 = 0.9701. v3 params were worse than v2 on AUC.

- **Narrow ±20% search** (Sub 22): Current config. Search space centered on v2 best.
  Result: 0.9798 — best submission.

- **Narrow ±20% resumed** (Sub 23): Resumed same study, 200 more trials. OOF improved marginally
  but leaderboard dropped to 0.9772. Same neighborhood.

- **Narrow ±10% fresh study** (Sub 24): Tighter bounds, fresh study. Overpredicted ponds 351→361.
  Score: 0.9650. Over-narrowing caused overfitting to training pond distribution.

**Lesson:** Two rounds of global search to find the neighborhood, then one round of ±20% local
refinement. Do not tighten further. ±10% is over-constrained.

---

## 9. Complete Submission History (26 Submissions)

All scores are on the Zindi leaderboard (test period generalization).

| # | Branch / Config | Score | AUC | F1 | Pred Ponds | Key Change | Lesson |
|---|---|---|---|---|---|---|---|
| 1 | Default params, 169 feat, balanced | 0.9468 | 0.9755 | 0.9277 | 350 | Baseline | Near-balanced; class_weight="balanced" slightly miscalibrates boundary |
| 2 | Optuna v1, 61 feat, balanced | 0.9458 | 0.9788 | 0.9237 | — | SHAP filter + Optuna simultaneously | Two changes at once undiagnosable; regression was from SHAP filter |
| 3 | Optuna v1, 169 feat, balanced | 0.9493 | 0.9819 | 0.9277 | — | SHAP filter removed, Optuna params kept | SHAP filter was the regression cause, not Optuna params |
| 4 | Optuna v1, 169 feat, no weight | 0.9523 | 0.9834 | 0.9316 | 352 | class_weight=None | Near-balanced dataset; balanced weighting miscalibrates boundary cases |
| 5 | + water_sar_stability interaction | 0.9468 | 0.9815 | 0.9237 | — | New interaction feature | Feature added noise; VV std already captured by VV__cv |
| 6 | + re-tuned Optuna (170 feat) | 0.9468 | 0.9815 | 0.9237 | — | Optuna on polluted feature set | Optuna params tuned with bad feature = bad params; remove bad feature first |
| 7 | Stacking ensemble | 0.9469 | 0.9697 | 0.9316 | 350 | LightGBM + RF + LogReg meta-learner | Meta-learner overfit on 963 samples; RF and LR are weaker learners |
| 8 | Prob avg sub4 + ensemble | 0.9518 | 0.9821 | 0.9316 | — | Simple probability averaging | Marginal gain; same 59 locations misclassified by all models |
| 9 | Temporal stability features | 0.9491 | 0.9813 | 0.9277 | — | Consecutive-change + cross-index agreement | F1 regressed; features added before Optuna re-tune |
| 10 | Pseudo-label 1 round (0.95/0.05) | 0.9528 | 0.9845 | 0.9316 | 356 | Semi-supervised high-confidence predictions | Small AUC gain; F1 unchanged — hard cases not in high-confidence zone |
| 11 | Pseudo-label 3 rounds | 0.9499 | 0.9772 | 0.9316 | 355 | Iterative pseudo-labeling | Round 2–3 inject borderline = noise; 1 round is maximum |
| 12 | NDTI + re1_nir, no pseudo | 0.9585 | 0.9853 | 0.9407 | 356 | Turbidity index + red edge ratio | First real F1 breakthrough; NDTI distinguishes turbid aquaculture water |
| 13 | NDTI + re1_nir + pseudo (stale) | 0.9557 | 0.9901 | 0.9328 | 363 | Pseudo on pre-NDTI model labels | Stale pseudo-labels from pre-NDTI model polluted NDTI signal |
| 14 | Optuna v2, 204 feat | 0.9720 | 0.9947 | 0.9569 | 352 | Expanded Optuna search with n_estimators tuned | colsample_bytree=0.407 forces feature diversity; min_child_samples=73 controls region-0 overfitting |
| 15 | Optuna v2 + pseudo | 0.9633 | 0.9912 | 0.9447 | 357 | Pseudo on Optuna v2 model | Pseudo consistently degrades F1; confirmed harmful pattern |
| 16 | Optuna v3, 204 feat (broken final fit) | 0.9658 | 0.9913 | 0.9487 | — | Wider search, n_estimators bug in train.py | Bug: final fit used early-stopping mean instead of Optuna n_estimators — undertrained |
| 17 | Optuna v3, 205 feat + subcluster | 0.9536 | 0.9917 | 0.9283 | 365 | Hardcoded subcluster centroids | Visual centroid estimates were wrong; introduced spatial noise |
| 18 | Optuna v3, fixed final fit | 0.9701 | 0.9898 | 0.9569 | 352 | Fixed n_estimators bug in train.py | Bug confirmed; v3 params still worse than v2 on AUC |
| 19 | Optuna v2, 203 feat, no dist_to_centroid | 0.9720 | 0.9948 | 0.9569 | 351 | Removed dist_to_pond_centroid (was rank 1 SHAP) | Spatial memorization removed; OOF dropped 0.001 but LB unchanged |
| 20 | SHAP filter 80 feat | 0.9660 | 0.9913 | 0.9492 | 355 | Top-80 SHAP filter applied | Optical raw bands carry distributed real signal; filter lost it |
| 21 | + GRU sequence model | 0.9651 | 0.9891 | 0.9492 | 355 | GRU OOF prob as additional feature | GRU OOF F1=0.9744 < LGBM F1=0.9821; weaker signal adds noise not resolution |
| **22** | **Narrow Optuna r1, 203 feat** | **0.9798** | **0.9949** | **0.9697** | **351** | **±20% search around Optuna v2 best** | **BEST. Local refinement found better minimum than global search.** |
| 23 | Narrow Optuna r2, resumed study | 0.9772 | 0.9948 | 0.9655 | 353 | Resumed Optuna study, 200 more trials | Better OOF but worse LB; same param neighborhood |
| 24 | Narrow Optuna r3, tighter bounds | 0.9650 | 0.9947 | 0.9451 | 361 | ±10% bounds, fresh study | Over-predicted ponds 351→361; params overfitting to training pond distribution |
| 25 | Prob avg sub22 + sub19 | 0.9748 | 0.9949 | 0.9614 | — | Probability averaging of two best | Sub19 errors are subset of sub22 errors; averaging dilutes, doesn't resolve |
| 26 | v6.1 baseline, 247 feat, Sub22 params | 0.9789 | 0.9923 | 0.9700 | 352 | New indices NDWI2+SAR_RVI+SABI+CI | F1 improved +0.000260 vs Sub22, AUC dropped -0.002587. colsample=0.373 was tuned for 203 feat, gives 92/tree at 247 feat (too many). Net tiny regression. |
| 27 | v7, 295 feat, Optuna re-tuned (n_est=893, colsample=0.339) | 0.9599 | 0.9923 | 0.9700 | 356 | +48 quarterly agg features (Q1–Q4 mean+max for NDWI, MNDWI, NDTI, re1_nir, SAR_RVI, SABI) | OOF=0.9908 but LB=0.9599 — OOF-to-LB gap of 0.031 (3× the typical 0.011). Quarterly features overfit to the training time period; they do not transfer to the test period. DO NOT use quarterly aggregations. |
| 28 | v6.1, 247 feat, Optuna v6.1 (n_est=900, colsample=0.264) | 0.9702 | 0.9946 | 0.9540 | 363 | Fresh Optuna sweep on 247 features, colsample settled at 0.264 (65/tree) | OOF-to-LB gap 0.020 (2× normal). 363 ponds = highest ever, worst F1 since Sub 18. colsample=0.264 is below the optimal floor of ~76/tree. Tuner minimises OOF variance by restricting features, but this causes over-prediction on the unseen test period. |
| 29 | v6.1, 247 feat, Optuna v6.1b (n_est=756, colsample=0.329) | 0.9716 | 0.9921 | 0.9580 | 360 | Second Optuna sweep with colsample floor 0.295 enforced (73–91/tree range) | 360 ponds, LB=0.9716. Floor enforcement reduced pond count 363→360 but still above 355 danger zone. Two Optuna rounds on v6.1 both over-predict. Sub 26 (Sub22 params directly) at 352 ponds/0.9789 is the v6.1 ceiling. Optuna tuning of v6.1 is CLOSED. |
| 30 | v6.2, 274 feat, Sub22 params (colsample=0.373→102/tree) | 0.9739 | 0.9922 | 0.9617 | 358 | +27 features: seasonal shape (peak_month, trough_month, amplitude) ×7 indices + consec_change for NDWI2/SAR_RVI | 358 ponds despite Sub22 params. Fixed colsample (0.373) gives 102/tree at 274 feat vs 76/tree at 203 feat. Colsample drift confirmed: each +44 feat with fixed colsample adds ~6 pond predictions. Fold 2 F1 dropped 0.993→0.975. OOF 0.9879. |
| 31 | v6.2, 274 feat, Sub22 params + proportional colsample (0.278→76/tree) | 0.9790 | 0.9925 | 0.9700 | 353 | Same as Sub30 but colsample=76/274=0.278 (proportional rescale) | 353 ponds (safe zone). Fold 2 recovered 0.9747→0.9811. OOF 0.9882 pre-cal. LB=0.9790 — new best on 274 features. Seasonal shape features marginally help vs Sub26 (0.9789→0.9790). Sub22 still best at 0.9798 by 0.0008. |

**Failed run (not numbered — submitted by mistake):** v6 tuned (256 feat): LB=0.9765, ponds=356. Optuna found fold3 F1=1.0 (overfit on 256-feat contaminated study). Tuned params gave over-prediction. This is the run that introduced the contaminated `optuna_study_v6.db` (see Lesson 14).

---

## 10. Approaches That Were Tried But Did Not Help

### 10.1 Stacking Ensemble (Sub 7)

Base learners: LightGBM, Random Forest, Logistic Regression.
Meta-learner: Logistic Regression on OOF base learner predictions.
Result: 0.9469 (worse than LightGBM alone at 0.9523).
Reason: 963 samples is too small for stacking. The meta-learner overfit to the base learner
ensemble, and the weaker base learners (RF, LR) dragged down the ensemble.
Source: `pipelines/training/ensemble.py` and `pipelines/training/base_learners.py` (still in the codebase).

### 10.2 Pseudo-Labeling (Subs 10, 11, 13, 15)

Mechanism: High-confidence test predictions (prob > 0.95 → label=1, prob < 0.05 → label=0) are
added to the training set. The model is retrained on the expanded set.

Single-round (Sub 10): Small AUC gain (+0.0007), no F1 gain. The hard cases remain unclassified.
Three rounds iterative (Sub 11): Degraded. Rounds 2–3 pull in borderline predictions as pseudo-labels.
After NDTI (Sub 13, 15): Consistently hurt. The pseudo-labels were generated by a pre-NDTI model
and were stale — they polluted the NDTI signal.

**Conclusion:** Pseudo-labeling with thresholds 0.95/0.05 adds no value for this problem. Hard cases
that are borderline in the model are not in the high-confidence zone and never receive pseudo-labels.
One round maximum if you insist on trying it, and always regenerate from the current best model.

### 10.3 GRU Sequence Model (Sub 21)

Described in detail in Section 11. Result: GRU OOF F1 = 0.9744, which is LOWER than LightGBM OOF
F1 = 0.9821. Adding the GRU probability as feature 204 to LightGBM produced Sub 21 = 0.9651, worse
than Sub 22 = 0.9798 (without GRU).

The weaker GRU signal adds noise, not resolution. The 59 misclassified samples in Sub 22 are also
misclassified by the GRU.

### 10.4 water_sar_stability Interaction Feature (Sub 5)

A handcrafted feature combining optical water signal with SAR temporal stability.
Result: regression. VV__cv already captures this. Adding correlated redundant features increases
colsample collision probability and dilutes tree diversity.

### 10.5 Subcluster Centroids (Sub 17)

Attempted to split the high-density region (region 1) into subclusters using visually estimated
centroid coordinates. The visual estimates were inaccurate. Introduced spatial noise.
Score dropped from 0.9701 to 0.9536. Do not attempt this without computing centroids from data.

### 10.6 Probability Averaging (Subs 8, 25)

Averaging the probabilities of two submissions.
Sub 8 (avg of sub4 + ensemble): 0.9518 — marginal vs sub4's 0.9523.
Sub 25 (avg of sub22 + sub19): 0.9748 — worse than sub22's 0.9798.
The errors of sub19 are a subset of sub22's errors. Averaging dilutes correct high-confidence
predictions rather than resolving the genuinely hard cases.

---

## 11. GRU Sequence Model

**Entry point:** `python -m pipelines.training.sequence_model`
**Source:** `pipelines/training/sequence_model.py` (currently untracked in git, on branch `sequence-model`)

### 11.1 Architecture

- Input: (batch, 12, 5) tensor — 12 timesteps, 5 channels
- Model: `PondGRU` — single-layer GRU, hidden_size=32, dropout=0.3, linear head to sigmoid output
- Parameters: ~5,000 total (appropriate for 963 training samples)
- Loss: BCELoss
- Optimizer: Adam, lr=3e-3, weight_decay=1e-4
- Training: 80 epochs, batch_size=64

### 11.2 Channels (5 selected indices)

Recomputed directly from raw bands (independent of the feature engineering pipeline):
1. NDWI
2. MNDWI
3. NDTI
4. VV (raw dB)
5. SAR_diff_db (VH - VV)

### 11.3 Training Discipline

- 5-fold CV on training data using the same splits as LightGBM (label × region strata)
- Per-fold normalization: stats computed on train split only, applied to val split (no leakage)
- OOF predictions: val fold probabilities from the fold's held-out GRU
- Final GRU: trained on full training data using full-train normalization stats
- Test inference: normalized by full-train stats, predicted by final GRU

### 11.4 Integration with LightGBM

The GRU OOF probability (`gru_prob`) is injected into both train and test feature parquets
as an additional column (feature 204). LightGBM is then retrained on 204 features.

### 11.5 Performance and Decision

- GRU standalone OOF: F1=0.9744, AUC not recorded
- LightGBM standalone OOF: F1=0.9821
- GRU-augmented LightGBM (Sub 21): Leaderboard = 0.9651

The GRU is weaker than LightGBM on the OOF. Adding a weaker signal as a feature introduces noise.
The 59 samples that LightGBM misclassifies are also misclassified by the GRU — it adds no
orthogonal information for the hard cases.

**Architecture rationale (why GRU and not CNN or Transformer):**
- GRU handles variable-length sequences and respects temporal order naturally
- Transformer attention needs substantially more data (963 samples is too few)
- 1D CNN ignores temporal ordering — month 3 is not adjacent to month 9
- GRU with hidden_size=32 has ~5k parameters — a sensible scale for 963 samples

---

## 12. SHAP Analysis

**Entry point:** `python -m pipelines.evaluation.shap_analysis`
**Source:** `pipelines/evaluation/shap_analysis.py`

SHAP values are computed on the final trained LightGBM model. Output:
- `outputs/evaluation/shap_importance.csv` — ranked list of features by mean |SHAP|
- SHAP bar plots

### 12.1 SHAP Feature Group Breakdown (Final v5 Model, 203 features)

| Group | Count | SHAP Share | Key features |
|---|---|---|---|
| Water indices (NDWI, MNDWI, AWEInsh) | 36 | 23.4% | NDWI__max, AWEInsh__cv, MNDWI__cv |
| SAR (VH, VV, SAR_diff_db) | 31 | 16.6% | SAR_diff_db__mean, VH__median, VV__median |
| Turbidity + algae (NDTI, re1_nir) | 24 | ~18% | NDTI__min, NDTI__mean_consec_change, re1_nir__max |
| Optical raw bands | 90 | 32.8% | Distributed; no single dominant feature |
| Vegetation (NDVI, NDRE) | 19 | 6.8% | NDVI__cv, NDVI__min |
| Spatial | 1 | 1.5% | region |
| Persistence counts | 5 | ~1% | NDWI_pos_count, MNDWI_pos_count |

**Important:** The 90 optical raw band features account for 32.8% of total SHAP importance despite
having no single dominant feature. This distributed signal means a SHAP filter that discards
lower-ranked raw band features loses real predictive information. Sub 20 (SHAP filter to 80 features)
confirmed this — score dropped from 0.9720 to 0.9660.

---

## 13. Key Technical Lessons Extracted from 31 Submissions

1. **One change per submission is non-negotiable.** Subs 2 and 5–6 were undiagnosable because
   multiple things changed. Every regression that took more than one submission to diagnose
   violated this rule.

2. **OOF consistently overestimates leaderboard by 0.011–0.031 points** due to temporal
   distribution shift. Higher OOF does not guarantee higher leaderboard. Optuna rounds 2 and 3
   and narrow r3 all found higher OOF but lower leaderboard scores. Sub 27 produced the worst
   gap seen (0.031) — quarterly features amplified temporal overfitting.

3. **SHAP filter from model A cannot be applied to model B.** Happened twice (Subs 2 and 20).
   Both caused regression. The filter must always be recomputed on the model it will be applied to.

4. **class_weight="balanced" is wrong for near-balanced data.** At 40.8% positive rate, the
   weight multiplier is 1.22x. This pushes borderline probabilities above 0.5, increasing false
   positives. Sub 4 gain: +0.004 F1 from removing it.

5. **dist_to_pond_centroid was spatial memorization, not signal.** Rank 1 SHAP before removal.
   After removal, OOF dropped 0.001, leaderboard unchanged (Subs 14 vs 19).

6. **NDTI is the single most impactful feature addition.** Sub 12: +0.009 F1, +0.0085 combined
   score from NDTI alone. Physical basis is clear and strong (turbid aquaculture water vs clear
   water bodies).

7. **Pseudo-labeling: one round maximum.** Three rounds degraded performance. After NDTI, pseudo
   consistently hurt because labels from the pre-NDTI model were stale. Always regenerate pseudo-
   labels from the current best model.

8. **Narrow Optuna beats global Optuna.** Global searches found the neighborhood. ±20% local
   refinement found the better minimum (0.9798 vs 0.9720). ±10% over-constrained and overfit.

9. **colsample_bytree is the most important hyperparameter.** Reducing from ~1.0 to 0.373 forces
   feature diversity per tree. With 203 features, high colsample makes all trees converge to
   NDWI-heavy splits. When the feature count grows (e.g., 247 in v6.1 or 295 in v7), colsample
   needs re-tuning: 0.373 × 247 ≈ 92 features/tree, 0.339 × 295 ≈ 100 features/tree — both
   too many and defeat the diversity objective.

10. **The n_estimators bug in train.py.** Final fit used the early-stopping mean iteration count
    × 1.05 instead of the Optuna n_estimators. This was fixed between Sub 16 and Sub 18.
    Sub 16 score (broken): 0.9658. Sub 18 score (fixed): 0.9701. The final model MUST use the
    Optuna n_estimators (870 in Sub 22, 893 in Sub 27) for the final fit on full data.

11. **Stacking fails at 963 samples.** Meta-learner overfit. Trust LightGBM alone.

12. **Probability averaging does not resolve hard cases.** The same ~59 samples are wrong in all
    model variants tried. Averaging dilutes confident correct predictions rather than fixing errors.

13. **SHAP filter auto-trigger is a silent danger.** The build_features pipeline auto-applies a
    SHAP filter when `outputs/evaluation/shap_importance.csv` exists. After adding new features and
    running SHAP on a different model, the old SHAP file nearly caused a stale-filter regression
    (same mistake as Subs 2 and 20). Protocol: always rename or delete `shap_importance.csv` before
    rebuilding features with any model change. Only recompute SHAP after training the new model.

14. **Optuna study contamination across feature versions.** When `tune.py` uses `load_if_exists=True`
    with the same study name/DB, trials from different feature sets get mixed. The v6 study
    (`optuna_study_v6.db`) was contaminated: trial #106 (best) was evaluated on 256 features, but
    subsequent 200 trials ran on 247 features. Mixed-feature-set studies produce unreliable param
    rankings. Fix: always use a new study name AND new DB file for each new feature version.

15. **Same Optuna basin found repeatedly.** The v6 tuner (200 trials) and the v6.1 tuner (200 more
    trials on contaminated DB) both converged on the same best trial #106: n_est=944, lr=0.1206,
    leaves=91, depth=6, colsample=0.275. Two independent runs finding the same params suggests a
    genuine local minimum for this feature space, but it is NOT better than Sub 22 on the leaderboard.

16. **Quarterly aggregations cause temporal overfitting. Do not use.** Sub 27 added Q1–Q4 mean and
    max for 6 indices (48 features, v7 = 295 total). OOF was 0.9908 (best ever), but leaderboard
    dropped to 0.9599 — a 0.031 OOF-to-LB gap (3× the typical 0.011). Physical explanation:
    quarterly statistics encode the absolute seasonal calendar of the training time period. The test
    period is a different time window, so Q1 Jan–Mar in train ≠ Q1 Jan–Mar in test in terms of
    agricultural state. Annual aggregations (mean, std, etc.) are time-period invariant; quarterly
    splits are not. The correct way to capture seasonal transitions is through consecutive-change
    stability features (already in the pipeline), not fixed quarterly bins.

17. **colsample has a floor AND Optuna on 247 features over-predicts. v6.1 Optuna is closed.**
    Sub 28: colsample=0.264 (65/tree) → 363 ponds, LB=0.9702.
    Sub 29: colsample=0.329 (81/tree, floor enforced) → 360 ponds, LB=0.9716.
    Both rounds of Optuna on 247 features produced pond counts above the 355 danger zone.
    The best v6.1 result was Sub 26 — Sub 22 params applied directly (no Optuna) → 352 ponds,
    LB=0.9789. The Optuna landscape at 247 features converges to higher leaves/lower min_child
    (more complex trees) that systematically over-predict on the test period.
    **Rule:** The colsample floor is 76/n_features. The v6.1 Optuna ceiling is 0.9716.
    The v6.1 ceiling applying Sub22 params directly is 0.9789. Sub22 (v5, 203 feat) at 0.9798
    remains the best overall. Do not run further Optuna sweeps on 247 features.

18. **Colsample must be proportionally rescaled when adding features (76/tree is the target density).**
    Sub 30 used Sub22 params (colsample=0.373) on 274 features → 102/tree → 358 ponds, LB=0.9739.
    Sub 31 used colsample=76/274=0.278 on the same 274 features → 76/tree → 353 ponds, LB=0.9790.
    Rescaling colsample recovered the pond count (358→353) and boosted LB by +0.0051.
    **Rule for any new feature set of size N:** set colsample = 76/N before the first training run.
    Do not carry over a colsample value tuned for a different feature count.
    Sub 31 (274 feat, 0.9790) is just 0.0008 behind Sub22 (203 feat, 0.9798). Seasonal shape
    features marginally helped vs v6.1 (0.9789→0.9790). The 76/tree density rule is validated.

---

## 14. OOF Hard Case Analysis (v6.1 model)

**Source:** `pipelines/evaluation/hard_case_analysis.py`

The v6.1 model (247 features, Sub 22 params) has **13 wrong OOF predictions**:
- TP=384, TN=566, FP=4, FN=9

### Error Groups by Probability Margin

| Group | N | Prob | Margin | Pattern |
|---|---|---|---|---|
| Near-boundary FN | 3 | 0.375 | 0.125 | Ponds just below 0.5 |
| Borderline FP+FN | 6 | 0.667 | 0.167 | Both types near boundary |
| Confident FP | 3 | 0.900 | 0.400 | Non-ponds with strong pond-like signals |
| Confident FN | 4 | 0.024–0.048 | 0.45+ | Ponds with very atypical spectral signatures |

### Most Discriminating Features (hard vs correct)

1. `re1_nir__max_consec_change` (0.887 normalized diff)
2. `re1_nir__mean_consec_change` (0.753)
3. `region` (0.671)
4. `NDWI__max_consec_change` (0.658)
5. `NDTI__monotone_fraction` (0.626)

### FP Signature (false alarms, n=4)

- All in region 1 (high-density zone)
- `MNDWI__cv` = 2.53 vs TN mean 0.36 — extreme optical variability
- These are seasonal wetlands or rice paddies that look like ponds in certain months

### FN Signature (missed ponds, n=9)

- `water_index_unanimous` = 0.630 vs TP mean 0.327
- `SAR_RVI__mean` = 0.501 vs TP mean 0.367
- These ponds were likely drained/harvested during part of the training period, giving them unusual temporal patterns

### Implication

The hard cases have temporal structure that the current aggregations do not fully capture:
- **FPs** are seasonally variable (wet some months, dry others)
- **FNs** are ponds with atypical seasonal management (draining cycles)

Quarter-level aggregations are the motivated fix: they would capture within-year seasonal transitions that month-level statistics collapse.

---

## 15. Current Repository State (git status at time of context generation)

**Branch:** `explore/new-ideas`

**Modified tracked files:**
- `.gitignore` — updated to exclude data/processed and outputs (not committed)
- `README.md` — updated with cold-start guide, reproduction steps, docs pointers (not committed)
- `pipelines/training/tune.py` — updated to narrow ±20% Optuna search (not committed)

**Untracked files (not yet staged):**
- `docs/EXPERIMENT_LOG.md` — full experiment log (all 27 submissions, key learnings, feature timeline, final config)
- `docs/MODEL_SCORECARD.md` — single-page model identity, performance, feature summary, training procedure
- `pipelines/training/sequence_model.py` — GRU model (experimental, Sub 21 artifact)
- `pipelines/training/tests/test_sequence_model.py` — GRU unit tests

**Commit history on `explore/new-ideas` (most recent first):**
1. `63c9e34` — feat(eval): hard case analysis script
2. `5b13ca9` — feat(features): v6.1 - drop CDOM index (SHAP noise), keep NDWI2+SAR_RVI+SABI+CI
3. `41315d0` — fix(tune): match class_weight=None to train.py, fresh v6 study name
4. `c628593` — feat(features): add 5 physics-motivated indices for v6 feature set

**Prior commits (main branch lineage):**
5. `00b0e16` — feat(pipeline): enhance training robustness and expand hyperparameter search
6. `78f2801` — feat: add NDTI/re1-NIR indices, stability features, and expanded tuning
7. `9d79d93` — test: implement pipeline integrity smoke test and refine documentation
8. `5bda16f` — feat: implement iterative refinement cycle and automated tuning
9. `70771c5` — feat: initial project setup for GeoAI Aquaculture Pond Identification

---

## 16. Pipeline File Map

```
aquaculture-pond-detection/
│
├── contracts/
│   └── schema.py              # Column names, band lists, MONTHS list, DataSchema validator
│
├── data/
│   ├── raw/
│   │   ├── Train.csv          # 963 rows × 147 cols (not committed)
│   │   └── Test.csv           # 858 rows × 146 cols (not committed)
│   └── processed/
│       ├── train_features.parquet   # 963 × 296 (295 features + ID + label)
│       └── test_features.parquet    # 858 × 296 (295 features + ID)
│
├── docs/
│   ├── EXPERIMENT_LOG.md      # All 27 submissions, key learnings, feature timeline
│   └── MODEL_SCORECARD.md     # Final model identity and performance card
│
├── experiments/
│   └── logs/
│       ├── optuna_study.db    # Resumable SQLite Optuna study
│       └── optuna_results.csv # All trial results
│
├── outputs/
│   ├── eda/                   # EDA plots
│   ├── features/
│   │   ├── feature_pipeline_meta.json
│   │   ├── feature_names.txt
│   │   └── region_kmeans.joblib
│   ├── models/
│   │   ├── lgbm_model.joblib
│   │   ├── calibrator.joblib
│   │   ├── best_params.json   # Written by tune.py, read by train.py
│   │   ├── oof_predictions.csv
│   │   ├── cv_summary.csv
│   │   ├── calibration_summary.csv
│   │   ├── gru_final.pt       # GRU weights (if sequence_model.py was run)
│   │   └── gru_oof_probs.csv  # GRU OOF probabilities
│   ├── evaluation/
│   │   └── shap_importance.csv  # Feature importance rankings
│   └── submissions/
│       └── submission.csv     # Zindi-ready output
│
├── pipelines/
│   ├── eda/
│   │   ├── run_eda.py
│   │   ├── class_balance.py
│   │   ├── missing_values.py
│   │   ├── regional_analysis.py
│   │   ├── spectral_separation.py
│   │   └── temporal_profiles.py
│   ├── features/
│   │   ├── build_features.py  # Main entry point
│   │   ├── indices.py         # Spectral index functions (pure)
│   │   └── aggregations.py    # Temporal aggregation + feature matrix builder
│   ├── training/
│   │   ├── train.py           # Main training entry point (528 lines)
│   │   ├── tune.py            # Optuna sweep
│   │   ├── cv_strategy.py     # 5-fold stratified CV on label × region
│   │   ├── calibration.py     # Isotonic regression calibrator
│   │   ├── sequence_model.py  # GRU model (experimental, untracked)
│   │   ├── ensemble.py        # Stacking ensemble (tried in Sub 7, not in use)
│   │   └── base_learners.py   # RF and LR base learners for stacking
│   └── evaluation/
│       ├── shap_analysis.py      # SHAP importance computation and plots
│       ├── hard_case_analysis.py # OOF hard case analysis (v6.1+)
│       └── metrics.py            # combined_score function (0.6*F1 + 0.4*AUC)
│
├── requirements.txt
├── pyproject.toml
└── .pre-commit-config.yaml
```

---

## 17. Reproduction Commands

To reproduce Sub 22 (best submission, 0.9798):

```bash
# Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Place raw data at:
#   data/raw/Train.csv
#   data/raw/Test.csv

# Build 203-feature matrix (v5)
python -m pipelines.features.build_features

# Narrow Optuna tuning (200 trials, ±20% around Sub 22 best)
python -m pipelines.training.tune

# Train final model + generate submission
python -m pipelines.training.train

# Output: outputs/submissions/submission.csv
```

Optional steps:
```bash
# Run EDA
python -m pipelines.eda.run_eda

# Run SHAP analysis (requires trained model)
python -m pipelines.evaluation.shap_analysis

# Run GRU sequence model (experimental, adds gru_prob to parquets)
python -m pipelines.training.sequence_model
# Then retrain LightGBM on 204 features:
python -m pipelines.training.train
```

---

## 18. Known Open Problems

 1. **The same ~59 test samples are consistently misclassified** across all model variants.
    No attempted intervention (pseudo-labeling, stacking, GRU, averaging, quarterly features) has
    resolved them. These are genuinely hard cases — seasonal wetlands (FPs) and ponds with
    unusual draining-cycle temporal patterns (FNs).

 2. **OOF vs leaderboard gap is ~0.011 under normal conditions; pathological features can widen it
    to 0.031.** The temporal distribution shift is real and structural. Quarterly aggregations
    (Sub 27) proved this gap can be catastrophically widened by features that encode the absolute
    seasonal calendar of the training period. Stick to temporally invariant aggregations.

 3. **The tune.py class_weight inconsistency.** Tuning uses `class_weight="balanced"` but
    train.py overrides it to `None`. Sub 22 was produced under this configuration so the
    best_params.json was tuned with a slightly different objective than the final model.
    Whether fixing this would improve or harm results is unknown.

 4. **Quarter-level aggregations are ruled out** (Sub 27 confirmed). OOF analysis (Section 14)
    correctly identified that hard cases have temporal structure not captured by current
    aggregations. However, the fix is NOT fixed quarterly bins — it must be time-period invariant.
    Motivated alternatives: (a) consecutive-change features on more indices, (b) within-year
    peak/trough timing features expressed as relative offsets rather than fixed calendar quarters,
    (c) seasonal amplitude features (summer max minus winter min).

 5. **GRU adds no value at current OOF level.** The GRU needs to outperform LightGBM on the
    OOF to add orthogonal signal. At F1=0.9744 vs 0.9821, it is worse. Possible future
    directions: larger hidden_size, more channels, attention mechanism — but data size limits
    what is feasible.

 6. **No spatial generalization test.** The project only measures temporal generalization
    (train period A → test period B). The two regions are both in the training set. A true
    geographic generalization test (train on region 1, test on region 0) has not been done.

 7. **Optuna study contamination risk.** The `optuna_study_v6.db` contains mixed trials from
    256-feature (v6) and 247-feature (v6.1) runs. For the next tune run, use study name
    `lgbm_aquaculture_v6_1` and a fresh DB `optuna_study_v6_1.db`. Any v7 tune run must use
    `lgbm_aquaculture_v7` and `optuna_study_v7.db` — but v7 is now a dead end (see Lesson 16).

 8. **Next feature engineering direction.** Temporally invariant seasonal features: peak-month
    index (argmax of NDWI over 12 months), amplitude (max-min for key indices), and consecutive
    stability features on additional indices. These capture seasonal timing without encoding
    absolute calendar position.

---

## 19. Tech Stack

| Layer | Library | Version constraint |
|---|---|---|
| Python | — | 3.10+ |
| Model | LightGBM | — |
| Tuning | Optuna | — |
| Interpretability | SHAP | — |
| Sequence model | PyTorch | — |
| CV + calibration | scikit-learn | — |
| Feature store | Pandas + PyArrow (Parquet) | — |
| Analytics | Matplotlib, Seaborn | — |
| Persistence | joblib | — |
| Testing | pytest | — |
| Linting | ruff | — |
| Pre-commit | pre-commit | — |
