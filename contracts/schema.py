"""
Single source of truth for column names, band definitions, and data contracts.
Every pipeline imports from here. Never hardcode band names elsewhere.
"""

from dataclasses import dataclass
from typing import ClassVar

MONTHS = [f"{i:02d}" for i in range(1, 13)]  # ["01", "02", ..., "12"]

SAR_BANDS = ["VH", "VV"]

OPTICAL_BANDS = [
    "blue",   # B2  ~490nm
    "green",  # B3  ~560nm
    "red",    # B4  ~665nm
    "re1",    # B5  ~705nm  red edge 1
    "re2",    # B6  ~740nm  red edge 2
    "re3",    # B7  ~783nm  red edge 3
    "nir",    # B8  ~842nm  broad NIR
    "nira",   # B8A ~865nm  narrow NIR
    "swir1",  # B11 ~1610nm
    "swir2",  # B12 ~2190nm
]

ALL_BANDS = SAR_BANDS + OPTICAL_BANDS  # 12 bands total

# Derived spectral indices — defined here so features/ and eda/ stay in sync
SPECTRAL_INDICES = [
    "NDWI",    # (green - nir)   / (green + nir)
    "MNDWI",   # (green - swir1) / (green + swir1)
    "NDVI",    # (nir   - red)   / (nir   + red)
    "NDRE",    # (nira  - re1)   / (nira  + re1)
    "AWEInsh", # 4*(green-swir1) - (0.25*nir + 2.75*swir2)
    "SAR_ratio",  # VH / VV
    "SAR_diff",   # VH - VV
]

# Column name generators
def raw_col(band: str, month: str) -> str:
    """e.g. raw_col('VH', '01') → 'VH_01'"""
    return f"{band}_{month}"

def all_raw_cols() -> list[str]:
    return [raw_col(b, m) for b in ALL_BANDS for m in MONTHS]

# Metadata columns
META_COLS = ["ID"]
TARGET_COL = "label"
WINDOW_METADATA_COLS = [
    "window_start", "window_length", "window_center",
    "window_start_sin", "window_start_cos",
    "window_center_sin", "window_center_cos"
]

# Temporal aggregation suffixes applied to every band and index
AGG_SUFFIXES = [
    "mean", "median", "std", "min", "max",
    "p10", "p90",          # 10th and 90th percentile
    "cv",                  # coefficient of variation = std/mean
]

# Persistence threshold counts — separate from scalar aggs
PERSISTENCE_FEATURES = [
    "NDWI_pos_count",      # months where NDWI > 0
    "NDVI_low_count",      # months where NDVI < 0.1
    "MNDWI_pos_count",     # months where MNDWI > 0
]


@dataclass
class DataSchema:
    """
    Runtime contract for a loaded dataframe.
    Call DataSchema.validate(df) before any pipeline step.
    """
    # required_raw_cols: ClassVar[list[str]] = field(default_factory=all_raw_cols)
    required_raw_cols: ClassVar[list] = []

    @staticmethod
    def validate_train(df) -> None:
        _check_columns(df, META_COLS + [TARGET_COL] + all_raw_cols())
        assert df[TARGET_COL].isin([0, 1]).all(), "label must be binary 0/1"
        assert df["ID"].is_unique, "ID must be unique"

    @staticmethod
    def validate_test(df) -> None:
        _check_columns(df, META_COLS + all_raw_cols())
        assert df["ID"].is_unique, "ID must be unique"


def _check_columns(df, expected: list[str]) -> None:
    missing = set(expected) - set(df.columns)
    extra   = set(df.columns) - set(expected)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    # Extra columns are a warning, not an error
    if extra:
        print(f"[schema] WARNING: unexpected columns present: {sorted(extra)}")