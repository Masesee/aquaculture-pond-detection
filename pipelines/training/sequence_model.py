"""
GRU-based temporal sequence model for aquaculture pond detection.

Operates on the 12-month time series of 5 key indices:
  NDWI, MNDWI, NDTI, VV (dB), SAR_diff_db

Each location becomes a (12, 5) tensor — 12 timesteps, 5 channels.
A one-layer GRU with hidden_size=32 reads the sequence and outputs
a single pond probability.

This probability (gru_prob) is appended to both train and test feature
parquets as feature 204. LightGBM is then retrained on all 204 features.

Why GRU over CNN or Transformer:
  - GRU handles variable-length sequences and captures order naturally
  - 963 samples is too small for a Transformer (attention needs data)
  - 1D CNN ignores temporal ordering — month 3 is not adjacent to month 9
  - GRU with hidden_size=32 has ~5k parameters — appropriate for 963 samples

Training discipline:
  - 5-fold CV on training data — OOF predictions only, no leakage
  - GRU is retrained from scratch on each fold's train split
  - Val split of each fold is predicted by the fold's held-out GRU
  - Final GRU is trained on full training data for test set inference

Run with:
    python -m pipelines.training.sequence_model

Outputs:
    outputs/models/gru_final.pt           ← final GRU weights
    outputs/models/gru_oof_probs.csv      ← OOF probabilities for inspection
    data/processed/train_features.parquet ← updated with gru_prob column
    data/processed/test_features.parquet  ← updated with gru_prob column
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score

from contracts.schema import TARGET_COL, MONTHS
from pipelines.training.cv_strategy import make_cv_splits
from pipelines.evaluation.metrics import combined_score

PROCESSED_DIR = ROOT / "data"    / "processed"
MODELS_DIR    = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Sequence configuration ────────────────────────────────────────────────────

# The 5 channels fed to the GRU — selected by SHAP as most discriminative
# and confirmed as temporally invariant signals
SEQUENCE_CHANNELS = ["NDWI", "MNDWI", "NDTI", "VV", "SAR_diff_db"]

# Raw column name patterns for each channel across 12 months
# NDWI, MNDWI, NDTI are computed from raw bands — we recompute them here
# VV and SAR_diff_db come directly from the raw band columns
# This keeps the sequence model independent of the feature pipeline

# ── GRU model ────────────────────────────────────────────────────────────────

class PondGRU(nn.Module):
    """
    Single-layer GRU for pond/non-pond classification from monthly sequences.

    Input:  (batch, 12, n_channels) float tensor
    Output: (batch,) float tensor of pond probabilities
    """

    def __init__(
        self,
        n_channels: int = 5,
        hidden_size: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = n_channels,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 12, n_channels)
        _, h_n = self.gru(x)          # h_n: (1, batch, hidden_size)
        h = h_n.squeeze(0)            # (batch, hidden_size)
        h = self.dropout(h)
        logit = self.head(h).squeeze(-1)   # (batch,)
        return torch.sigmoid(logit)


# ── Sequence feature extraction ───────────────────────────────────────────────

def _extract_sequences(df: pd.DataFrame) -> np.ndarray:
    """
    Builds (n_samples, 12, 5) array of monthly index values.

    Recomputes indices from raw bands to stay independent of the
    feature engineering pipeline. Uses the same formulas as indices.py.
    """
    EPS = 1e-9
    n   = len(df)
    seq = np.zeros((n, 12, len(SEQUENCE_CHANNELS)), dtype=np.float32)

    for t, month in enumerate(MONTHS):
        green  = df[f"green_{month}"].values.astype(np.float32)
        nir    = df[f"nir_{month}"].values.astype(np.float32)
        swir1  = df[f"swir1_{month}"].values.astype(np.float32)
        red    = df[f"red_{month}"].values.astype(np.float32)
        vh     = df[f"VH_{month}"].values.astype(np.float32)
        vv     = df[f"VV_{month}"].values.astype(np.float32)

        ndwi        = (green - nir)   / (green + nir   + EPS)
        mndwi       = (green - swir1) / (green + swir1 + EPS)
        ndti        = (red   - green) / (red   + green + EPS)
        sar_diff_db = vh - vv

        seq[:, t, 0] = ndwi
        seq[:, t, 1] = mndwi
        seq[:, t, 2] = ndti
        seq[:, t, 3] = vv
        seq[:, t, 4] = sar_diff_db

    # Normalise each channel to zero mean, unit std across training samples
    # Normalisation statistics are computed inside the caller to avoid leakage
    return seq


def _normalise(
    seq_train: np.ndarray,
    seq_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits normalisation on train sequences, applies to both.
    Returns normalised arrays and the (mean, std) fit on train.
    Shape of seq: (n, 12, channels)
    Stats are computed over (n, 12) jointly per channel.
    """
    # Reshape to (n*12, channels) for stat computation
    n_tr, T, C = seq_train.shape
    flat_tr    = seq_train.reshape(-1, C)

    mu  = flat_tr.mean(axis=0)          # (C,)
    std = flat_tr.std(axis=0) + 1e-9    # (C,)

    seq_train_norm = (seq_train - mu) / std
    seq_test_norm  = (seq_test  - mu) / std

    return seq_train_norm, seq_test_norm, mu, std


# ── Training utilities ────────────────────────────────────────────────────────

def _train_gru(
    X_seq: np.ndarray,
    y:     np.ndarray,
    n_epochs:   int   = 80,
    batch_size: int   = 64,
    lr:         float = 3e-3,
    hidden_size: int  = 32,
    dropout:    float = 0.3,
    seed:       int   = 42,
) -> PondGRU:
    """
    Trains a PondGRU on the provided sequences and labels.
    Returns the trained model.
    """
    torch.manual_seed(seed)
    model     = PondGRU(
        n_channels  = X_seq.shape[2],
        hidden_size = hidden_size,
        dropout     = dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y,     dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def _predict_gru(model: PondGRU, X_seq: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_seq, dtype=torch.float32)
        probs = model(X_t).numpy()
    return probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load raw data (not feature matrices — GRU works from raw bands) ───
    print("=== Loading raw data ===")
    train_raw = pd.read_csv(ROOT / "data" / "raw" / "Train.csv")
    test_raw  = pd.read_csv(ROOT / "data" / "raw" / "Test.csv")
    print(f"  Train: {train_raw.shape} | Test: {test_raw.shape}")

    y_train = train_raw[TARGET_COL].values

    # ── Extract sequences ─────────────────────────────────────────────────
    print("=== Extracting sequences ===")
    seq_train = _extract_sequences(train_raw)   # (963, 12, 5)
    seq_test  = _extract_sequences(test_raw)    # (858, 12, 5)
    print(f"  seq_train: {seq_train.shape} | seq_test: {seq_test.shape}")

    # ── Normalise on full train stats (CV normalisation done per fold below) ─
    # Full-train normalisation used only for the final model fit
    seq_train_norm_full, seq_test_norm_full, _, _ = _normalise(
        seq_train, seq_test
    )

    # ── 5-fold OOF loop ───────────────────────────────────────────────────
    print("\n=== 5-fold GRU OOF ===")

    # Load feature parquet to get the split structure (region column needed)
    train_feats = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    splits      = make_cv_splits(train_feats, n_splits=5, random_state=42)

    oof_probs   = np.zeros(len(train_raw), dtype=np.float32)

    for fold, (tr_pos, val_pos) in enumerate(splits):
        # Normalise using only the fold's train split to prevent leakage
        seq_tr_raw  = seq_train[tr_pos]
        seq_val_raw = seq_train[val_pos]
        y_tr        = y_train[tr_pos]
        y_val       = y_train[val_pos]

        seq_tr_norm, seq_val_norm, _, _ = _normalise(seq_tr_raw, seq_val_raw)

        model = _train_gru(seq_tr_norm, y_tr)
        fold_probs = _predict_gru(model, seq_val_norm)
        oof_probs[val_pos] = fold_probs

        fold_preds = (fold_probs >= 0.5).astype(int)
        f1  = f1_score(y_val, fold_preds)
        auc = roc_auc_score(y_val, fold_probs)
        print(f"  Fold {fold}: F1={f1:.4f} | AUC={auc:.4f} | "
              f"Score={combined_score(f1, auc):.4f}")

    # OOF aggregate
    oof_preds = (oof_probs >= 0.5).astype(int)
    oof_f1    = f1_score(y_train, oof_preds)
    oof_auc   = roc_auc_score(y_train, oof_probs)
    print(f"  OOF: F1={oof_f1:.4f} | AUC={oof_auc:.4f} | "
          f"Score={combined_score(oof_f1, oof_auc):.4f}")

    # Save OOF for inspection
    oof_df = pd.DataFrame({
        "ID":       train_raw["ID"].values,
        "label":    y_train,
        "gru_prob": oof_probs,
    })
    oof_df.to_csv(MODELS_DIR / "gru_oof_probs.csv", index=False)
    print("  Saved: outputs/models/gru_oof_probs.csv")

    # ── Train final GRU on full training data ─────────────────────────────
    print("\n=== Training final GRU on full training data ===")
    final_gru = _train_gru(seq_train_norm_full, y_train)
    test_gru_probs = _predict_gru(final_gru, seq_test_norm_full)

    print(f"  Test GRU prob range: "
          f"[{test_gru_probs.min():.3f}, {test_gru_probs.max():.3f}]")
    print(f"  Test GRU positive rate: {(test_gru_probs >= 0.5).mean():.3f}")

    # Save final GRU
    torch.save(final_gru.state_dict(), MODELS_DIR / "gru_final.pt")
    print("  Saved: outputs/models/gru_final.pt")

    # ── Inject gru_prob into feature parquets ─────────────────────────────
    print("\n=== Injecting gru_prob into feature parquets ===")
    train_feats = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_feats  = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    # Align on ID to be safe
    train_id_to_prob = dict(zip(train_raw["ID"].values, oof_probs))
    test_id_to_prob  = dict(zip(test_raw["ID"].values,  test_gru_probs))

    train_feats["gru_prob"] = train_feats["ID"].map(train_id_to_prob)
    test_feats["gru_prob"]  = test_feats["ID"].map(test_id_to_prob)

    # Validate no NaN was introduced
    assert not train_feats["gru_prob"].isna().any(), \
        "NaN in train gru_prob — ID alignment failed"
    assert not test_feats["gru_prob"].isna().any(), \
        "NaN in test gru_prob — ID alignment failed"

    train_feats.to_parquet(PROCESSED_DIR / "train_features.parquet", index=False)
    test_feats.to_parquet(PROCESSED_DIR  / "test_features.parquet",  index=False)

    print(f"  Train parquet updated: {train_feats.shape}")
    print(f"  Test  parquet updated: {test_feats.shape}")
    print("\n=== Sequence model complete ===")
    print("  Next: python -m pipelines.training.train")


if __name__ == "__main__":
    main()