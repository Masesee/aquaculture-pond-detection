# GeoAI Aquaculture Pond Identification

**Zindi competition** — binary classification of 10 m satellite pixels as aquaculture pond or other land cover,
using 12 months of Sentinel-1 SAR + Sentinel-2 optical data.

**Best leaderboard score: 0.9798** (0.6 × F1 + 0.4 × AUC) | F1: 0.9697 | AUC: 0.9949

> Full performance breakdown → [`docs/MODEL_SCORECARD.md`](docs/MODEL_SCORECARD.md)
> Full experiment history (25 submissions, lessons learned) → [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)

---

## What is this?

A modular, production-ready ML pipeline that detects aquaculture ponds from tabular remote sensing data.
Each sample is a geographic point with 12 monthly observations across 11 satellite bands
(Sentinel-1 VH/VV + Sentinel-2 B02–B8A). The model must generalise from one time period (train) to a
different time period (test), making temporal robustness the primary challenge.

**Geography:** Two regions (~48°E, 39–40°N)
**Dataset:** 963 training samples (40.8% ponds) | 858 test samples
**Metric:** `0.6 × F1 + 0.4 × AUC` at a fixed 0.5 decision threshold

---

## How to reproduce the best submission (Sub 22)

```bash
# 1. Install
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 2. Place raw data
#    data/raw/Train.csv  and  data/raw/Test.csv  must exist

# 3. Build features (203-feature v5 set)
python -m pipelines.features.build_features

# 4. Run narrow Optuna tuning  (+/-20% search around Sub 22 best params, 200 trials)
python -m pipelines.training.tune

# 5. Train final model + generate submission
python -m pipelines.training.train
```

Output lands in `outputs/submissions/submission.csv`.

**Exact hyperparameters used for Sub 22** (automatically written by `tune.py` and read by `train.py`):

| Parameter | Value |
|---|---:|
| n_estimators | 870 |
| learning_rate | 0.10764 |
| num_leaves | 102 |
| max_depth | 8 |
| min_child_samples | 85 |
| subsample | 0.6474 |
| colsample_bytree | 0.3732 |
| reg_alpha | 1.30e-4 |
| reg_lambda | 1.32e-3 |
| class_weight | None |

> See [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) for why each parameter matters
> and the full progression from baseline (0.9468) to best (0.9798).

---

## Pipeline architecture

Each stage is an independent entry point. Run them in order for the full refinement cycle.

| Stage | Command | Outputs |
|---|---|---|
| 1 - EDA | `python -m pipelines.eda.run_eda` | `outputs/eda/` |
| 2 - Feature engineering | `python -m pipelines.features.build_features` | `data/processed/`, `outputs/features/` |
| 3 - Hyperparameter tuning | `python -m pipelines.training.tune` | `outputs/models/best_params.json`, `experiments/logs/` |
| 4 - Train + inference | `python -m pipelines.training.train` | `outputs/models/`, `outputs/submissions/` |
| 5 - SHAP evaluation | `python -m pipelines.evaluation.shap_analysis` | `outputs/evaluation/` |

### What the feature pipeline does

- **Regional clustering:** KMeans(k=2) on lon/lat splits the dataset into a low-density NW region
  (321 samples, 4% ponds) and a high-density SE region (642 samples, 59% ponds).
  Region indicator is the only spatial feature kept in the final model.
- **Spectral indices computed:** NDWI, MNDWI, NDVI, NDRE, AWEInsh, NDTI (turbidity), re1/NIR ratio,
  SAR_diff_db (VV minus VH in dB).
- **Temporal aggregation per band/index:** mean, median, std, min, max, p10, p90, cv, range (9 stats).
- **Temporal stability per selected index:** max_consec_change, mean_consec_change, monotone_fraction.
- **Total features:** 203 (v5 final set).
- **Optional SHAP filter:** reduces to top-N features; run after a first SHAP analysis pass.

### What the training pipeline does

1. 5-fold stratified CV on a `label x region` interaction stratum.
2. LightGBM with early stopping (50 rounds patience) per fold.
3. Isotonic regression calibrator fitted on out-of-fold probabilities.
4. Final refit on full training data using `n_estimators` from Optuna (not the early-stopping mean).
5. Calibrated probabilities -> threshold at 0.5 -> `submission.csv`.

---

## Project structure

```text
aquaculture-pond-detection/
├── contracts/              # Band definitions and data schemas
├── data/
│   ├── raw/                # Train.csv, Test.csv  (not committed)
│   └── processed/          # Parquet feature matrices (not committed)
├── docs/
│   ├── EXPERIMENT_LOG.md   # All 25 submissions with scores and lessons
│   └── MODEL_SCORECARD.md  # Final model identity, performance, and config
├── experiments/
│   └── logs/               # Optuna SQLite study DB and trial logs
├── outputs/
│   ├── eda/                # Class balance, spectral separation plots
│   ├── features/           # Feature metadata, region KMeans model
│   ├── models/             # Trained LightGBM folds, calibrators, best_params.json
│   ├── evaluation/         # SHAP importance rankings and bar plots
│   └── submissions/        # submission.csv (Zindi-ready)
├── pipelines/
│   ├── eda/                # EDA module + tests
│   ├── features/           # Feature engineering module + tests
│   ├── training/           # train.py, tune.py, sequence_model.py + tests
│   └── evaluation/         # SHAP analysis module + tests
├── requirements.txt
└── pyproject.toml
```

---

## Experiment history and model card

All 25 submissions are documented with scores, the single change tested, and the lesson learned:
→ [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)

Final model identity, feature breakdown, known limitations, and training procedure:
→ [`docs/MODEL_SCORECARD.md`](docs/MODEL_SCORECARD.md)

---

## Running tests

```bash
pytest
```

Gate tests are deterministic and should complete in under 2 seconds each.
They cover data schema validation, single-batch overfit checks, and pipeline contract integrity.

---

## Tech stack

| Layer | Library |
|---|---|
| Model | LightGBM |
| Tuning | Optuna |
| Interpretability | SHAP |
| Feature store | Pandas + Parquet |
| CV + calibration | scikit-learn |
| Sequence model (experimental) | PyTorch GRU |
| Analytics | Matplotlib, Seaborn |

---

## License

MIT — see [LICENSE](LICENSE).
