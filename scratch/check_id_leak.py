import pandas as pd
from pathlib import Path

ROOT = Path("D:/Data Science/GeoAI Aquaculture Pond Identification/aquaculture-pond-detection")
train_df = pd.read_csv(ROOT / "data/raw/Train.csv")
test_df = pd.read_csv(ROOT / "data/raw/Test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nFirst 10 Train IDs:")
print(train_df["ID"].head(10).tolist())

# Check if there are base IDs and how they are structured
base_ids = train_df["ID"].apply(lambda x: x.split("_w")[0])
print("\nTrain unique base IDs:", base_ids.nunique())
print("Value counts of base IDs in Train:")
print(base_ids.value_counts().head(10))
