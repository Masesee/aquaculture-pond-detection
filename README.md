# GeoAI Aquaculture Pond Identification

A modular, production-ready pipeline for identifying aquaculture ponds using multi-modal remote sensing data (Sentinel-1 SAR and Sentinel-2 Optical).

## Project Overview

This project aims to detect aquaculture ponds by leveraging monthly temporal profiles of spectral indices and radar backscatter. It employs a regional clustering approach to handle spatial distribution shifts and implements probability calibration for reliable binary classification.

### Key Features
- **Regional Clustering:** Uses K-Means on coordinates to handle spatial heterogeneity (e.g., high-density vs. low-density pond regions).
- **Multi-Modal Features:** Combines Sentinel-1 (VH, VV) and Sentinel-2 (10 bands) across 12 months.
- **Spectral Indices:** Calculates NDWI, MNDWI, NDVI, NDRE, AWEInsh, and SAR ratios.
- **Temporal Aggregation:** Computes 9 statistical aggregations (mean, std, percentiles, etc.) per band/index.
- **Probability Calibration:** Uses Isotonic Regression on out-of-fold (OOF) predictions to refine classification thresholds.

---

## 🏗 Architecture & Workflow

The pipeline is split into three main stages, each with its own entry point:

### 1. Exploratory Data Analysis (EDA)
Generates comprehensive visualisations and statistical reports on class balance, missing values, regional distributions, and spectral separation.
- **Entry point:** `python -m pipelines.eda.run_eda`
- **Outputs:** `outputs/eda/`

### 2. Feature Engineering
Transforms raw monthly band data into a high-dimensional feature matrix (169 features). It fits a regional model on training coordinates and calculates temporal aggregations.
- **Entry point:** `python -m pipelines.features.build_features`
- **Outputs:** `data/processed/` (Parquet files), `outputs/features/`

### 3. Model Training & Inference
Runs 5-fold stratified CV (label × region), trains a LightGBM classifier, performs isotonic calibration, and generates the final submission.
- **Entry point:** `python -m pipelines.training.train`
- **Outputs:** `outputs/models/` (Models and OOF data), `outputs/submissions/`

### 4. Evaluation & Interpretability
Calculates detailed metrics and generates SHAP-based feature importance plots to understand model decisions.
- **Entry point:** `python -m pipelines.evaluation.shap_analysis`
- **Outputs:** `outputs/evaluation/`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Git](https://git-scm.com/)

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

### Running the Pipeline
Execute the stages in order:
```bash
# 1. Run EDA (Optional)
python -m pipelines.eda.run_eda

# 2. Build features
python -m pipelines.features.build_features

# 3. Train model and generate submission
python -m pipelines.training.train

# 4. Run evaluation
python -m pipelines.evaluation.shap_analysis
```

### Running Tests
The project uses `pytest` for unit testing.
```bash
pytest
```

---

## 📁 Directory Structure

```text
├── contracts/          # Data schemas and band definitions
├── data/
│   ├── raw/            # Original Train.csv and Test.csv
│   └── processed/      # Parquet feature matrices
├── outputs/
│   ├── eda/            # Analysis plots and summaries
│   ├── features/       # Feature metadata and region models
│   ├── models/         # Trained LightGBM and Calibrators
│   ├── evaluation/     # SHAP plots and metric reports
│   └── submissions/    # Final Zindi-ready submission.csv
├── pipelines/
│   ├── eda/            # EDA logic + tests
│   ├── features/       # Feature engineering modules + tests
│   ├── training/       # Training, CV, and Calibration logic + tests
│   └── evaluation/     # Metrics and SHAP analysis + tests
```

---

## 🛠 Tech Stack
- **Language:** Python 3.10
- **Model:** LightGBM
- **Feature Store:** Pandas + Parquet
- **Analytics:** Scikit-learn, Matplotlib, Seaborn
- **Documentation:** Markdown

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
