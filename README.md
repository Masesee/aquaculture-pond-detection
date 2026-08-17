# GeoAI Aquaculture Pond Identification

**Zindi competition** — binary classification of 10 m satellite pixels as aquaculture pond or other land cover,
using 12 months of Sentinel-1 SAR + Sentinel-2 optical data.

**Phase 1 — Full-Year Observations (archived): 0.9798** (0.6 × F1 + 0.4 × AUC) | F1: 0.9697 | AUC: 0.9949
**Phase 2 — Masked Temporal Windows (current): 0.87796** | F1: 0.8608 | AUC: 0.9038

> Full performance breakdown for both phases → [`docs/MODEL_SCORECARD.md`](docs/MODEL_SCORECARD.md)
> Full experiment history (84+ submissions, lessons learned) → [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
> Revised challenge context and adaptation strategy → [`docs/REVISED_CHALLENGE_CONTEXT.md`](docs/REVISED_CHALLENGE_CONTEXT.md)

---

## What is this?

A modular, production-ready ML pipeline that detects aquaculture ponds from tabular remote sensing data.
Each sample is a geographic point with 12 monthly observations across **12 satellite bands**
(Sentinel-1 VH/VV + Sentinel-2 B02–B8A, including all 3 red-edge bands). The model must generalise from one time period (train) to a
different time period (test), making **temporal robustness** the primary challenge.

The competition ran in two phases:

| Phase | Train | Test | Coordinates | Test Temporal Availability | Best Score |
|---|---|---|---|---|---|
| **1** (archived) | 963 (40.8% ponds) | 858 | Available | Full 12 months | **0.9798** |
| **2** (current) | 1,821 | 1,030 | **Removed** | 4–6 consecutive months only | **0.87796** |

**Metric:** `0.6 × F1 + 0.4 × AUC` at a fixed 0.5 decision threshold

---

## How to reproduce (Phase 2 — current)

```bash
# 1. Install
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 2. Place raw data
#    data/raw/Train.csv  and  data/raw/Test.csv  must exist (Phase 2: 1,821 / 1,030 samples)

# 3. Build features (232-feature set with mask augmentation)
python -m pipelines.features.build_features

# (Optional) Run KS drift pruning to obtain 146 invariant features
python -m pipelines.features.adversarial_pruning

# 4. (Optional) Inject GRU temporal sequence features
python -m pipelines.training.sequence_model

# 5. Run Optuna tuning (100 trials)
python -m pipelines.training.tune

# 6. Train LightGBM + generate seed-42 probs
python -m pipelines.training.train

# 7. Train XGBoost + generate seed-42 probs
python -m pipelines.training.train_xgb

# 8. Train CatBoost + generate seed-42 probs
python -m pipelines.training.train_catboost

# 9. Blend into final Triad ensemble submission
python -m pipelines.training.blend_ensemble
```

Output lands in `outputs/submissions/submission.csv`.

---

### How to reproduce Phase 1 best (Sub 22, archived)

```bash
# Use Phase 1 data (963 / 858 samples, with coordinates)
python -m pipelines.features.build_features
python -m pipelines.training.tune      # narrow ±20% Optuna around v2 best
python -m pipelines.training.train
```

Phase 1 used coordinate-based KMeans region features and `dist_to_pond_centroid`
(both removed in Phase 2 when coordinates were dropped from the competition data).

---

## Pipeline architecture

Each stage is an independent entry point. Run them in order for the full refinement cycle.

| Stage | Command | Outputs |
|---|---|---|
| 1 - EDA | `python -m pipelines.eda.run_eda` | `outputs/eda/` |
| 2 - Feature engineering | `python -m pipelines.features.build_features` | `data/processed/`, `outputs/features/` |
| 3 - KS drift pruning | `python -m pipelines.features.adversarial_pruning` | `outputs/features/invariant_features.txt` |
| 4 - GRU sequence injection | `python -m pipelines.training.sequence_model` | `outputs/models/gru_final.pt`, `data/processed/*.parquet` (+gru_prob) |
| 5 - Hyperparameter tuning | `python -m pipelines.training.tune` | `outputs/models/best_params.json`, `experiments/logs/` |
| 6 - LightGBM train | `python -m pipelines.training.train` | `outputs/models/lgbm_model.joblib`, `outputs/submissions/submission.csv` |
| 7 - XGBoost train | `python -m pipelines.training.train_xgb` | `outputs/models/xgb_model.joblib`, `outputs/submissions/` |
| 8 - CatBoost train | `python -m pipelines.training.train_catboost` | `outputs/models/cb_model.joblib`, `outputs/submissions/` |
| 9 - Triad ensemble blend | `python -m pipelines.training.blend_ensemble` | `outputs/submissions/submission.csv` |
| 10 - Stacking ensemble (alternative) | `python -m pipelines.training.ensemble` | `outputs/submissions/submission_ensemble.csv` |
| 11 - SHAP evaluation | `python -m pipelines.evaluation.shap_analysis` | `outputs/evaluation/` |

### What the feature pipeline does

- **Mask augmentation (Phase 2):** Each training sample is expanded to 24 copies,
  each with a different consecutive 4-, 5-, or 6-month observation window. All other
  months are set to NaN. This teaches the model to predict from partial-year data.
- **Spectral indices computed (10 per month):** NDWI, MNDWI, NDVI, NDRE, AWEInsh,
  NDTI (turbidity), re1/NIR ratio, SWI, NFAI, SAR_diff_db (VH minus VV in dB).
- **Temporal aggregation per band/index:** mean, median, std, min, max, p10, p90,
  cv, range (9 stats × 22 sources = 198 features).
- **Temporal sequence features:**
  - Persistence fractions (5): fraction of valid months exceeding index thresholds
  - Consecutive-change stats (15): max/mean monthly change, monotone fraction
  - Linear trend slopes (5): OLS slope over valid month indices
  - Cross-index agreement (2): water index consensus across NDWI/MNDWI/AWEInsh
- **Window metadata (7):** `window_start`, `window_length`, `window_center` +
  sin/cos encodings (critical for partial-year test samples).
- **Total features:** 232 (full set). Optionally reduced to top-80 by SHAP or
  146 by KS drift pruning.
- **Phase 1 differences:** KMeans region cluster (`region` indicator) and
  `dist_to_pond_centroid` were present; both removed when coordinates were dropped.

### What the training pipeline does

1. **Single-window StratifiedGroupKFold** (5 folds, grouped by original sample ID
   to prevent augmentation leakage).
2. **Triad ensemble** of LightGBM (leaf-wise), XGBoost (depth-wise), and CatBoost
   (oblivious trees) — each with independent training pipelines.
3. **Early stopping** (50 rounds) per fold for LGBM/XGB; default iterations for CatBoost.
4. **Isotonic regression calibrator** fitted on out-of-fold probabilities.
5. **Final refit** on full training data with **seed averaging** (3 seeds: 42, 100, 2026).
6. **Prior shift correction** (optional, blocked by default under Zindi rules).
7. **Equal 1/3 Triad probability blend** → threshold at 0.5 → `submission.csv`.
8. Alternative: **Stacking ensemble** with LGBM + RF + LogReg base learners and
   logistic regression meta-learner.
9. Alternative: **GRU feature injection** adds `gru_prob` column from a 5-channel
   (NDWI, MNDWI, NDTI, VV, SAR_diff_db) PyTorch GRU trained on raw 12-month sequences.

---

## Project structure

```text
aquaculture-pond-detection/
├── contracts/
│   └── schema.py              # Band definitions, indices, column contracts
├── data/
│   ├── raw/                   # Train.csv, Test.csv  (not committed)
│   └── processed/             # Parquet feature matrices (not committed)
├── docs/
│   ├── EXPERIMENT_LOG.md      # All 84+ submissions with scores and lessons
│   ├── REVISED_CHALLENGE_CONTEXT.md  # Phase 2 strategy and adaptation
│   └── MODEL_SCORECARD.md     # Final model identity, performance, config
├── experiments/
│   └── logs/                  # Optuna SQLite studies and trial logs
├── outputs/
│   ├── eda/                   # Class balance, spectral separation plots
│   ├── features/              # Feature metadata, invariant_features.txt
│   ├── models/                # Trained models, calibrators, best_params.json, OOF CSVs
│   ├── evaluation/            # SHAP importance rankings and bar plots
│   └── submissions/           # submission.csv (Zindi-ready)
├── pipelines/
│   ├── eda/                   # EDA: run_eda, compare_features, adversarial_validation + tests
│   ├── features/              # Feature engineering: build_features, aggregations,
│   │                          #   indices, adversarial_pruning + tests
│   ├── training/              # train.py, tune.py, train_xgb.py, train_catboost.py,
│   │                          #   sequence_model.py, ensemble.py, blend_ensemble.py,
│   │                          #   cv_strategy.py, calibration.py + tests
│   └── evaluation/            # SHAP analysis, metrics, oof_error_analysis + tests
├── requirements.txt
└── pyproject.toml
```

---

## Experiment history and model card

All 84+ submissions across both phases are documented with scores, the
single change tested, and the lesson learned:
→ [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)

Final model identity, feature breakdown, known limitations, and training procedure:
→ [`docs/MODEL_SCORECARD.md`](docs/MODEL_SCORECARD.md)

Phase 2 adaptation strategy (mask augmentation, KS pruning, Triad ensemble):
→ [`docs/REVISED_CHALLENGE_CONTEXT.md`](docs/REVISED_CHALLENGE_CONTEXT.md)

---

## Running tests

```bash
pytest
```

Gate tests are deterministic and should complete in under 2 seconds each.
They cover schema validation, index correctness, NaN propagation, CV integrity,
calibration, GRU, and ensemble logic — organised per-module in `pipelines/*/tests/`.

---

## Tech stack

| Layer | Library |
|---|---|
| **Tabular models** | LightGBM, XGBoost, CatBoost |
| **Tuning** | Optuna |
| **Interpretability** | SHAP |
| **Feature store** | Pandas + Parquet |
| **CV + calibration** | scikit-learn |
| **Sequence model** | PyTorch GRU |
| **Drift detection** | SciPy (KS test) |
| **Analytics** | Matplotlib, Seaborn |
| **Code quality** | Ruff |

---

## License

MIT — see [LICENSE](LICENSE).
