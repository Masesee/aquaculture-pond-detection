"""
EDA entrypoint. Run with:
    python -m pipelines.eda.run_eda

Produces all artefacts under outputs/eda/.
Every plot saved to disk — nothing interactive, reproducible headless.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from contracts.schema import DataSchema

from pipelines.eda.class_balance    import run_class_balance
from pipelines.eda.missing_values   import run_missing_values
from pipelines.eda.spectral_separation import run_spectral_separation
from pipelines.eda.temporal_profiles  import run_temporal_profiles

OUTPUT_DIR = ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ROOT / "data" / "raw" / "Train.csv"
TEST_PATH  = ROOT / "data" / "raw" / "Test.csv"


def main():
    print("=== Loading data ===")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # Validate contracts before anything else
    DataSchema.validate_train(train)
    DataSchema.validate_test(test)
    print(f"Train: {train.shape} | Test: {test.shape}")

    # Convert unobserved -9999 values to NaN immediately
    import numpy as np
    train = train.replace(-9999, np.nan).replace(-9999.0, np.nan)
    test  = test.replace(-9999, np.nan).replace(-9999.0, np.nan)

    print("\n=== Q1: Class balance ===")
    run_class_balance(train, OUTPUT_DIR)

    print("\n=== Q3: Missing values ===")
    run_missing_values(train, test, OUTPUT_DIR)

    print("\n=== Q4: Regional analysis (SKIPPED: coordinates removed) ===")
    # run_regional_analysis(train, test, OUTPUT_DIR)

    print("\n=== OOD check (SKIPPED: coordinates removed) ===")
    # from pipelines.eda.regional_analysis import run_ood_check
    # run_ood_check(train, test, OUTPUT_DIR)

    print("\n=== Q2/Q5: Spectral separation ===")
    run_spectral_separation(train, OUTPUT_DIR)

    print("\n=== Monthly temporal profiles ===")
    run_temporal_profiles(train, OUTPUT_DIR)

    print(f"\n=== EDA complete. Artefacts in {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()