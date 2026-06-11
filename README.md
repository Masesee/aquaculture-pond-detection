# GeoAI Aquaculture Pond Identification

A modular, production-ready pipeline for identifying aquaculture ponds using multi-modal remote sensing data (Sentinel-1 SAR and Sentinel-2 Optical).

## Project Overview

This project aims to detect aquaculture ponds by leveraging monthly temporal profiles of spectral indices and radar backscatter. It employs a regional clustering approach to handle spatial distribution shifts and implements probability calibration for reliable binary classification.

### Key Features
- **Regional Clustering:** Uses K-Means on coordinates to handle spatial heterogeneity (e.g., high-density vs. low-density pond regions).
- **Log-Transformed Spatial Features:** Computes distance to the primary pond cluster centroid using a log-transformation (log1p) to reduce the leverage of distant out-of-distribution points.
- **Multi-Modal Features:** Combines Sentinel-1 (VH, VV) and Sentinel-2 (10 bands) across 12 months.
- **Spectral Indices:** Calculates NDWI, MNDWI, NDVI, NDRE, AWEInsh, NDTI (Turbidity), and re1/NIR ratios.
- **Temporal Aggregation & Stability:** Computes 9 statistical aggregations (mean, std, percentiles, etc.) and temporal stability metrics (consecutive-month absolute change) for key indices.
- **Automated Feature Selection:** Optional SHAP-based filtering to reduce dimensionality to the top 60 most impactful features.
- **Hyperparameter Optimization:** Expanded Optuna sweep for LightGBM, including `n_estimators` tuning and broad search ranges.
- **Pseudo-Labeling:** Support for iterative pseudo-labeling to leverage high-confidence test predictions for model enhancement.
- **Probability Calibration:** Uses Isotonic Regression on out-of-fold (OOF) predictions to refine classification thresholds.

---

## Architecture and Workflow

The pipeline is split into several stages, each with its own entry point:

### 1. Exploratory Data Analysis (EDA)
Generates comprehensive visualisations and statistical reports on class balance, missing values, regional distributions, and spectral separation.
- **Entry point:** `python -m pipelines.eda.run_eda`
- **Outputs:** `outputs/eda/`

### 2. Feature Engineering
Transforms raw monthly band data into a high-dimensional feature matrix. It fits a regional model on training coordinates, calculates temporal aggregations, and optionally filters features based on prior SHAP importance.
- **Entry point:** `python -m pipelines.features.build_features`
- **Outputs:** `data/processed/`, `outputs/features/`

### 3. Hyperparameter Tuning
Performs an automated Optuna study to find optimal LightGBM parameters, maximizing a combined F1 and AUC score.
- **Entry point:** `python -m pipelines.training.tune`
- **Outputs:** `outputs/models/best_params.json`, `experiments/logs/`

### 4. Model Training and Inference
Runs 5-fold stratified CV, trains a LightGBM classifier (automatically loading tuned parameters if available), performs isotonic calibration, and generates the final submission.
- **Entry point:** `python -m pipelines.training.train`
- **Outputs:** `outputs/models/`, `outputs/submissions/`

### 5. Evaluation and Interpretability
Calculates detailed metrics and generates SHAP-based feature importance plots and CSV rankings.
- **Entry point:** `python -m pipelines.evaluation.shap_analysis`
- **Outputs:** `outputs/evaluation/`

---

## Getting Started

### Prerequisites
- Python 3.10+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd aquaculture-pond-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Full Pipeline Workflow (Refinement Cycle)
To achieve the best results, run the following sequence:

1. **Initial Build:** `python -m pipelines.features.build_features` (Builds full feature set)
2. **Initial Train:** `python -m pipelines.training.train` (Trains baseline model)
3. **Tune:** `python -m pipelines.training.tune` (Finds optimal hyperparameters)
4. **Analyze:** `python -m pipelines.evaluation.shap_analysis` (Generates importance ranking)
5. **Filtered Build:** `python -m pipelines.features.build_features` (Filters to top 60 features)
6. **Final Train:** `python -m pipelines.training.train` (Trains optimized compact model)

### Running Tests
The project uses `pytest` for unit testing.
```bash
pytest
```

---

## Directory Structure

```text
├── contracts/          # Data schemas and band definitions
├── data/
│   ├── raw/            # Original Train.csv and Test.csv
│   └── processed/      # Parquet feature matrices
├── experiments/
│   └── logs/           # Optuna study DB and logs
├── outputs/
│   ├── eda/            # Analysis plots and summaries
│   ├── features/       # Feature metadata and region models
│   ├── models/         # Trained LightGBM, Calibrators, and best_params.json
│   ├── evaluation/     # SHAP importance rankings and plots
│   └── submissions/    # Final Zindi-ready submission.csv
├── pipelines/
│   ├── eda/            # EDA logic + tests
│   ├── features/       # Feature engineering modules + tests
│   ├── training/       # Training, Tuning, CV, and Calibration logic + tests
│   └── evaluation/     # Metrics and SHAP analysis + tests
└── tests/              # End-to-end integration and pipeline tests
```

---

## Tech Stack
- **Language:** Python 3.10
- **Model:** LightGBM
- **Tuning:** Optuna
- **Interpretability:** SHAP
- **Feature Store:** Pandas + Parquet
- **Analytics:** Scikit-learn, Matplotlib, Seaborn
- **Documentation:** Markdown

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
