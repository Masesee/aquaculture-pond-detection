# Aquaculture Pond Detection — Experiment Log

Full record of every submission made during the competition, including
the reasoning behind each change, what it measured, and what it taught us.

---

## Baseline Configuration

| Item | Value |
|---|---|
| Dataset | Zindi aquaculture pond detection |
| Train samples | 963 |
| Test samples | 858 |
| Positive rate | 40.8% |
| Metric | 0.6 × F1 + 0.4 × AUC at fixed 0.5 threshold |
| Leaderboard #1 | 1.000 (likely test set memorization) |

---

## Submission History

| # | Branch / Config | Score | AUC | F1 | Predicted Ponds | Key Change | Lesson |
|---|---|---|---|---|---|---|---|
| 1 | Default params, 169 feat, balanced | 0.9468 | 0.9755 | 0.9277 | 350 | Baseline | Near-balanced classes; class_weight="balanced" slightly miscalibrates boundary |
| 2 | Optuna v1, 61 feat, balanced | 0.9458 | 0.9788 | 0.9237 | — | SHAP filter + Optuna simultaneously | Two changes at once undiagnosable; SHAP filter on 169-feat model applied to different model = regression |
| 3 | Optuna v1, 169 feat, balanced | 0.9493 | 0.9819 | 0.9277 | — | SHAP filter removed, Optuna params kept | SHAP filter was the regression cause, not Optuna params |
| 4 | Optuna v1, 169 feat, no weight | 0.9523 | 0.9834 | 0.9316 | 352 | class_weight=None | Near-balanced dataset; balanced weighting miscalibrates boundary cases |
| 5 | + water_sar_stability interaction | 0.9468 | 0.9815 | 0.9237 | — | New interaction feature | Feature added noise; VV std already captured by VV__cv |
| 6 | + re-tuned Optuna (170 feat) | 0.9468 | 0.9815 | 0.9237 | — | Optuna on polluted feature set | Optuna params tuned with bad feature = bad params; remove bad feature first |
| 7 | Stacking ensemble | 0.9469 | 0.9697 | 0.9316 | 350 | LightGBM + RF + LogReg meta-learner | Meta-learner overfit on 963 samples; trusted weaker learners (RF, LR) over LGBM |
| 8 | Prob avg sub4 + ensemble | 0.9518 | 0.9821 | 0.9316 | — | Simple probability averaging | Marginal; F1 identical, confirms same 59 locations misclassified by all models |
| 9 | Temporal stability features | 0.9491 | 0.9813 | 0.9277 | — | Consecutive-change + cross-index agreement | F1 regressed; features added before Optuna re-tune |
| 10 | Pseudo-label 1 round (0.95/0.05) | 0.9528 | 0.9845 | 0.9316 | 356 | Semi-supervised from high-confidence test predictions | Small AUC gain; F1 unchanged — hard cases not in high-confidence zone |
| 11 | Pseudo-label 3 rounds | 0.9499 | 0.9772 | 0.9316 | 355 | Iterative pseudo-labeling | Round 2-3 inject borderline predictions = noise; 1 round is the maximum |
| 12 | NDTI + re1_nir, no pseudo | 0.9585 | 0.9853 | 0.9407 | 356 | Turbidity index + red edge ratio | First real F1 breakthrough; NDTI distinguishes turbid aquaculture water from clear water |
| 13 | NDTI + re1_nir + pseudo (stale) | 0.9557 | 0.9901 | 0.9328 | 363 | Pseudo on pre-NDTI model labels | Stale pseudo-labels from pre-NDTI model polluted NDTI signal |
| 14 | Optuna v2, 204 feat | 0.9720 | 0.9947 | 0.9569 | 352 | Expanded Optuna search space with n_estimators tuned | colsample_bytree=0.407 forces feature diversity; min_child_samples=73 controls region-0 overfitting |
| 15 | Optuna v2 + pseudo | 0.9633 | 0.9912 | 0.9447 | 357 | Pseudo on Optuna v2 model | Pseudo consistently degrades F1; confirmed harmful pattern |
| 16 | Optuna v3, 204 feat (broken final fit) | 0.9658 | 0.9913 | 0.9487 | — | Wider search, but n_estimators bug | train.py used early-stopping mean iter instead of Optuna n_estimators — model undertrained |
| 17 | Optuna v3, 205 feat + subcluster | 0.9536 | 0.9917 | 0.9283 | 365 | Hardcoded subcluster centroids | Visual centroid estimates were wrong; introduced spatial noise |
| 18 | Optuna v3, fixed final fit | 0.9701 | 0.9898 | 0.9569 | 352 | Fixed n_estimators bug in train.py | Bug confirmed; v3 params still worse than v2 on AUC |
| 19 | Optuna v2, 203 feat, no dist_to_centroid | 0.9720 | 0.9948 | 0.9569 | 351 | Removed dist_to_pond_centroid (was rank 1 SHAP) | Spatial memorization removed; model forced to rely on spectral signal; OOF dropped slightly but LB held |
| 20 | SHAP filter 80 feat | 0.9660 | 0.9913 | 0.9492 | 355 | Top-80 SHAP filter applied | Optical raw bands at 30% group importance are not noise — distributed real signal lost |
| 21 | + GRU sequence model | 0.9651 | 0.9891 | 0.9492 | 355 | GRU OOF prob as additional feature | GRU OOF F1=0.9744 < LGBM F1=0.9821; weaker signal adds noise not resolution |
| **22** | **Narrow Optuna r1, 203 feat** | **0.9798** | **0.9949** | **0.9697** | **351** | ±20% search around Optuna v2 best | **Best submission. Local refinement found better minimum than global search.** |
| 23 | Narrow Optuna r2, resumed study | 0.9772 | 0.9948 | 0.9655 | 353 | Resumed Optuna study, 200 more trials | Marginally better OOF but worse LB; same param neighborhood |
| 24 | Narrow Optuna r3, tighter bounds | 0.9650 | 0.9947 | 0.9451 | 361 | ±10% bounds, fresh study | Over-predicted ponds 351→361; params overfitting to training pond distribution |
| 25 | Prob avg sub22 + sub19 | 0.9748 | 0.9949 | 0.9614 | — | Probability averaging of two best subs | Sub19 errors are subset of sub22 errors; averaging dilutes not improves |

---

## Key Learnings

### One change per submission is non-negotiable
Submissions 2 and 5-6 were undiagnosable because multiple things changed simultaneously.
Every regression in this project that took more than one submission to diagnose violated this rule.

### OOF score overestimates leaderboard performance
The temporal generalization gap (train period ≠ test period) means OOF consistently ran
0.03-0.05 points above leaderboard. Higher OOF does not guarantee higher leaderboard score.
Specifically: Optuna rounds 2, 3, and the narrow r3 all found higher OOF but lower leaderboard.

### SHAP filter computed on model A cannot be applied to model B
Applied twice (sub 2, sub 20). Both times it caused regression.
The filter must always be computed on the same model it will be applied to.

### class_weight="balanced" is wrong for near-balanced datasets
With 40.8% positive rate, balanced weighting gives ponds weight 1.22x.
This pushes borderline probabilities above 0.5, increasing false positives.
Sub 4 confirmed: removing it improved F1 by 0.004.

### Spatial memorization inflates OOF but hurts temporal generalization
dist_to_pond_centroid was rank 1 SHAP with value 1.548 before removal.
After removal OOF dropped 0.0012 but leaderboard score was identical (sub 19 = sub 14).
The spatial feature was memorizing training pond locations, not learning physics.

### NDTI is the single most impactful feature addition
Normalized Difference Turbidity Index = (red - green) / (red + green).
Aquaculture ponds have elevated red reflectance from biological load (fish waste, algae).
Clear water bodies (rivers, reservoirs) have lower red relative to green.
Sub 12 showed +0.0091 F1 gain from adding NDTI alone.

### Pseudo-labeling: one round only, never iterative
One round improved AUC slightly (sub 10). Three rounds degraded both metrics (sub 11).
After NDTI was added, pseudo consistently hurt because labels were generated by pre-NDTI models.
Always regenerate pseudo-labels from the current best model, never reuse stale ones.

### Narrow Optuna beats global Optuna
Global searches (subs 14, 16, 18) found good params but missed the local optimum.
The ±20% narrow search around v2 best found a better solution (sub 22: 0.9798 vs 0.9720).
The ±10% search overcorrected and overfit (sub 24).
Optimal: two rounds of global search, one round of ±20% local refinement.

### colsample_bytree is the most important hyperparameter for this problem
Reducing from 0.994 (default) to 0.373 (sub 22) forces feature diversity per tree.
With 203 features, high colsample means all trees converge to NDWI-heavy splits.
Low colsample creates an internal ensemble of diverse feature-subset models.

---

## Feature Engineering Timeline

| Version | Features | Key Additions |
|---|---|---|
| v1 | 144 raw | 12 bands × 12 months |
| v2 | 169 | + temporal aggregations (mean, std, cv, min, max, p10, p90, range) + indices (NDWI, MNDWI, NDVI, NDRE, AWEInsh, SAR_diff_db) + persistence counts + spatial |
| v3 | 180 | + consecutive-change features (NDWI, MNDWI, VV) + cross-index water agreement |
| v4 | 204 | + NDTI + re1_nir + their aggregations and change stats |
| **v5 (final)** | **203** | v4 minus dist_to_pond_centroid |

---

## Final Model Config

| Parameter | Value |
|---|---:|
| Features | 203 |
| Model | LightGBM (binary classifier) |
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
| CV | 5-fold stratified on label × region interaction |
| Calibration | Isotonic regression on OOF predictions |
| Final fit note | n_estimators=870 (Optuna value, not early-stopping mean) |

**Leaderboard: 0.9798 | AUC: 0.9949 | F1: 0.9697**