# Model Scorecard

Single-page reference for the final model's performance, configuration,
and known limitations across both competition phases. Intended for anyone
replicating or extending this work.

---

## Identity (Phase 2 — Current)

| Field | Value |
|---|---|
| Task | Binary classification: aquaculture pond vs other land cover |
| Geography | Central Asia (~48°E, 39–40°N) |
| Resolution | 10 m × 10 m per sample (single-pixel, no spatial context) |
| Train samples | 1,821 |
| Test samples | 1,030 |
| Test availability | 4–6 consecutive months of observations (remainder masked to -9999) |
| Coordinates | Removed from competition data |
| Competition | Zindi aquaculture pond detection (revised Phase 2) |
| Best submission | Sub 61 — Quantile Misalignment Fix (best performing) & Sub 64 — Tuned Raw Ensemble (best compliant) |

---

## Performance

### Phase 2 (Current)

**Best Leaderboard Performance (Sub 61 - Quantile Misalignment Fix):**

| Metric | OOF (train period) | Leaderboard (test period) |
|---|---|---|
| Combined score | 0.9828 | **0.8780** |
| F1 @ 0.5 threshold | — | **0.8608** |
| ROC-AUC | — | **0.9038** |
| Predicted ponds | — | 658 / 1,030 |

**Best Zindi-Compliant Pipeline (Sub 64 - Tuned Raw Ensemble):**

| Metric | OOF (train period) | Leaderboard (test period) |
|---|---|---|
| Combined score | 0.9818 | **TBD** |
| F1 @ 0.5 threshold | 0.9726 | **TBD** |
| ROC-AUC | 0.9955 | **TBD** |
| Predicted ponds | — | 616 / 1,030 |

OOF overestimates leaderboard by ~0.10-0.11 due to temporal distribution shift
between train and test periods (larger gap than Phase 1 because of 4–6 month masking).

### Phase 1 (Archived — Sub 22)

| Metric | OOF (train period) | Leaderboard (test period) |
|---|---|---|
| Combined score | 0.9908 | **0.9798** |
| F1 @ 0.5 threshold | 0.9847 | **0.9697** |
| ROC-AUC | 0.9986 | **0.9949** |
| Predicted ponds | — | 351 / 858 |

---

## Feature Summary (Phase 2, Full Set)

**Total features: 232 (full pipeline) | 146 (KS-pruned) | 100 (SHAP-selected, best)**

| Group | Count | Description |
|---|---|---|
| **Window metadata** | 7 | window_start, window_length, window_center + sin/cos encodings of start and center |
| **SAR raw band aggs** | 18 | VH, VV: mean, median, std, min, max, p10, p90, cv, range |
| **Optical raw band aggs** | 90 | 10 bands (blue, green, red, re1, re2, re3, nir, nira, swir1, swir2) × 9 stats |
| **Water index aggs** | 27 | NDWI, MNDWI, AWEInsh: 9 stats each |
| **Turbidity + algae aggs** | 18 | NDTI, re1_nir_ratio: 9 stats each |
| **Vegetation index aggs** | 18 | NDVI, NDRE: 9 stats each |
| **Additional spectral aggs** | 18 | SWI, NFAI, SAR_diff_db: 9 stats each |
| **Persistence fractions** | 5 | NDWI_pos, MNDWI_pos, NDVI_low, AWEInsh_pos, SAR_diff_neg15 (fraction of valid months) |
| **Consecutive-change stats** | 15 | NDWI, MNDWI, VV, NDTI, re1_nir: max_change, mean_change, monotone_fraction |
| **Linear trend slopes** | 5 | NDWI, MNDWI, NDTI, re1_nir, SAR_diff_db: OLS slope over valid months |
| **Cross-index agreement** | 2 | water_index_agreement, water_index_unanimous (NDWI+MNDWI+AWEInsh consensus) |

**Acronym reference:**
- **NDWI/MNDWI** — (Modified) Normalized Difference Water Index
- **NDVI/NDRE** — Normalized Difference Vegetation Index / Red Edge
- **AWEInsh** — Automated Water Extraction Index (no shadow)
- **NDTI** — Normalized Difference Turbidity Index (aquaculture-specific, most impactful Phase 1 addition)
- **SWI** — Sentinel-2 Water Index
- **NFAI** — Normalized Floating Algae Index
- **re1_nir_ratio** — Red Edge 1 / NIR chlorophyll proxy
- **SAR_diff_db** — VH - VV in dB (water specular signature)

**Aggregation stats (per band/index):** mean, median, std, min, max, p10, p90, cv, range

**Phase 1 features removed (coordinates dependent):**
- `region` KMeans binary indicator (removed with coordinates in Phase 2)
- `dist_to_pond_centroid` (removed Sub 19 — caused spatial memorisation)

---

## Model Configuration (Phase 2 Best: Sub 57)

### Triad Ensemble — LightGBM

| Parameter | Value |
|---|---|
| Objective | binary |
| Boosting | gbdt |
| n_estimators | 1000 (early stopping: 50 rounds) |
| learning_rate | 0.02 (default; Optuna search range 0.01–0.08) |
| num_leaves | 31 (Optuna range 15–45) |
| max_depth | -1 (Optuna range 3–6) |
| min_child_samples | 20 (Optuna range 100–300) |
| subsample | 0.8 (Optuna range 0.60–0.90) |
| colsample_bytree | 0.8 (Optuna range 0.30–0.70) |
| reg_alpha | 0.1 (Optuna range 0.01–10.0) |
| reg_lambda | 1.0 (Optuna range 0.1–50.0) |
| class_weight | None |

### Triad Ensemble — XGBoost

| Parameter | Value |
|---|---|
| Objective | binary:logistic |
| n_estimators | 1000 (early stopping: 50 rounds) |
| learning_rate | 0.03 |
| max_depth | 4 |
| min_child_weight | 5 |
| subsample | 0.70 |
| colsample_bytree | 0.50 |
| reg_alpha | 0.5 |
| reg_lambda | 5.0 |

### Triad Ensemble — CatBoost

| Parameter | Value |
|---|---|
| Objective | Logloss |
| iterations | 1000 |
| learning_rate | 0.03 |
| depth | 6 |
| l2_leaf_reg | 3.0 |
| random_seed | 42 |

### Ensemble Configuration

| Component | Setting |
|---|---|
| Blend type | Equal 1/3 probability blend (no meta-learner) |
| Seed averaging | 3 seeds per model (42, 100, 2026), predictions averaged before blend |
| Calibration | Isotonic regression per model on OOF, applied before blend |
| Prior shift | Optional post-blend correction; blocked by default (Zindi rule compliance) |
| GRU injection | Optional: PyTorch GRU (5→32→1, 5k params) on NDWI/MNDWI/NDTI/VV/SAR_diff_db sequences |

---

## Training Procedure (Phase 2)

1. **Load raw data** — `Train.csv` (1,821 samples) + `Test.csv` (1,030 samples). Replace -9999 with NaN.
2. **Mask augmentation** — Expand each training sample into 24 copies with consecutive 4/5/6-month windows (49,167 augmented rows).
3. **Build features** — 232 NaN-safe features via `pipelines/features/build_features.py`:
   - Window metadata, 10 spectral/SAR indices per month, temporal aggregations (9 stats × 22 sources), persistence fractions, consecutive-change stats, linear trend slopes, cross-index agreement.
4. **KS drift pruning (optional)** — Two-sample KS test per feature. Prune features with KS ≥ 0.20. Retain ~146 domain-invariant features. Output: `invariant_features.txt`.
5. **SHAP feature selection (optional)** — Run SHAP analysis, retain top-100 features + window metadata.
6. **5-fold StratifiedGroupKFold CV** — Groups by original sample ID (prevents augmentation leakage). Single-window validation subset.
7. **Train Triad models** — LightGBM, XGBoost, CatBoost independently. Early stopping (50 rounds) for LGBM/XGB. Seed averaging across [42, 100, 2026].
8. **Calibration** — Isotonic regression on OOF probabilities per model.
9. **Blend** — Equal 1/3 average of calibrated test probabilities.
10. **Prior shift correction (optional)** — Adjust for expected test set prior if known.
11. **Threshold at 0.5** → generate `submission.csv`.

---

## Known Limitations

1. **Temporal generalisation gap** — OOF scores consistently overestimate leaderboard performance by ~0.114 (Phase 2) / ~0.011 (Phase 1). Train and test periods are different time windows with different seasonal and environmental conditions.
2. **No spatial context** — Each pixel is classified independently. Neighbourhood features, texture, or image-patch context could provide additional signal (possible ceiling).
3. **Small dataset** — 1,821 training samples (Phase 2) limits model complexity. The GRU at ~5k parameters is near the upper bound of what this data supports.
4. **Manual feature design** — The 232-feature set is hand-crafted based on domain knowledge. Automated feature learning (e.g., CNN on time-frequency representations) might discover unexpected signals.
5. **Partial-year ambiguity** — Test samples with only 4–6 observation months cannot be distinguished from dry seasonal ponds that only exist 4–6 months per year. This is an irreducible ambiguity given the data.
6. **Calibration optimism** — The isotonic regressor is calibrated on OOF data that shares the same temporal period as training. When applied to a shifted test distribution, calibration may be imperfect.
7. **Competition rule constraints** — Prior shift correction is blocked by default and threshold tuning is not permitted, limiting the ability to compensate for known distribution shifts.

---

## Reproducibility

### Phase 2 (Current)
```bash
python -m pipelines.features.build_features
python -m pipelines.features.adversarial_pruning   # optional
python -m pipelines.training.tune                   # optional
python -m pipelines.training.train
python -m pipelines.training.train_xgb
python -m pipelines.training.train_catboost
python -m pipelines.training.blend_ensemble
```

### Phase 1 (Archived)
```bash
# Requires Phase 1 data (963 / 858 samples with coordinates)
python -m pipelines.features.build_features
python -m pipelines.training.tune
python -m pipelines.training.train
```
