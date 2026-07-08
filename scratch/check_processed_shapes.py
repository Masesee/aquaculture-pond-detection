import pandas as pd
from pathlib import Path

ROOT = Path("D:/Data Science/GeoAI Aquaculture Pond Identification/aquaculture-pond-detection")
train_features = pd.read_parquet(ROOT / "data/processed/train_features.parquet")

print("Processed train shape:", train_features.shape)
print("\nFirst 10 Processed Train IDs:")
print(train_features["ID"].head(10).tolist())

# Check how base IDs are split
base_ids = train_features["ID"].apply(lambda x: x.split("_w")[0])
print("\nUnique base IDs in processed train:", base_ids.nunique())
print("Value counts of base IDs:")
print(base_ids.value_counts().head(10))
