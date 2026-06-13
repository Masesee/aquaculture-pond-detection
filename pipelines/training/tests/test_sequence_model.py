"""
Gate tests for the GRU sequence model.
Deterministic. No real data. Must pass in < 5s.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.training.sequence_model import (
    PondGRU,
    _extract_sequences,
    _normalise,
    _train_gru,
    _predict_gru,
    MONTHS,
)
from contracts.schema import ALL_BANDS, raw_col


@pytest.fixture
def minimal_raw_df() -> pd.DataFrame:
    """12-row synthetic raw dataframe with all required band columns."""
    rng = np.random.default_rng(42)
    n   = 12
    data = {
        "ID":  [f"ID_TR_{i:04d}" for i in range(n)],
        "lon": rng.uniform(48.0, 49.5, n),
        "lat": rng.uniform(39.0, 40.5, n),
        "label": rng.integers(0, 2, n),
    }
    for band in ALL_BANDS:
        for month in MONTHS:
            if band in ["VH", "VV"]:
                data[raw_col(band, month)] = rng.uniform(-28, -8, n)
            else:
                data[raw_col(band, month)] = rng.uniform(300, 6000, n)
    return pd.DataFrame(data)


# ── PondGRU architecture tests ────────────────────────────────────────────────

def test_gru_output_shape():
    model = PondGRU(n_channels=5, hidden_size=32)
    x     = torch.randn(8, 12, 5)
    out   = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_gru_output_range():
    model = PondGRU(n_channels=5, hidden_size=32)
    x     = torch.randn(16, 12, 5)
    out   = model(x)
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_gru_no_nan_output():
    model = PondGRU(n_channels=5, hidden_size=32)
    x     = torch.randn(8, 12, 5)
    out   = model(x)
    assert not torch.isnan(out).any()


def test_gru_different_inputs_different_outputs():
    """Non-identical inputs must produce non-identical outputs."""
    torch.manual_seed(0)
    model = PondGRU(n_channels=5, hidden_size=32)
    model.eval()
    x1 = torch.zeros(4, 12, 5)
    x2 = torch.ones(4, 12, 5)
    with torch.no_grad():
        o1 = model(x1)
        o2 = model(x2)
    assert not torch.allclose(o1, o2), "Model produces identical output for different inputs"


# ── Sequence extraction tests ─────────────────────────────────────────────────

def test_extract_sequences_shape(minimal_raw_df):
    seq = _extract_sequences(minimal_raw_df)
    assert seq.shape == (12, 12, 5), f"Expected (12,12,5), got {seq.shape}"


def test_extract_sequences_no_nan(minimal_raw_df):
    seq = _extract_sequences(minimal_raw_df)
    assert not np.isnan(seq).any()


def test_extract_sequences_no_inf(minimal_raw_df):
    seq = _extract_sequences(minimal_raw_df)
    assert not np.isinf(seq).any()


def test_extract_sequences_ndwi_channel_range(minimal_raw_df):
    """NDWI (channel 0) must be in [-1, 1]."""
    seq = _extract_sequences(minimal_raw_df)
    ndwi_vals = seq[:, :, 0]
    assert ndwi_vals.min() >= -1 - 1e-5
    assert ndwi_vals.max() <=  1 + 1e-5


# ── Normalisation tests ───────────────────────────────────────────────────────

def test_normalise_train_zero_mean():
    """After normalisation, train channel means should be near zero."""
    rng   = np.random.default_rng(0)
    seq_tr = rng.standard_normal((50, 12, 5)).astype(np.float32) + 3.0
    seq_te = rng.standard_normal((20, 12, 5)).astype(np.float32) + 3.0
    norm_tr, _, _, _ = _normalise(seq_tr, seq_te)
    channel_means = norm_tr.reshape(-1, 5).mean(axis=0)
    assert np.allclose(channel_means, 0.0, atol=1e-5), \
        f"Train channel means not zero after normalisation: {channel_means}"


def test_normalise_shapes_preserved():
    rng   = np.random.default_rng(1)
    seq_tr = rng.standard_normal((40, 12, 5)).astype(np.float32)
    seq_te = rng.standard_normal((15, 12, 5)).astype(np.float32)
    norm_tr, norm_te, mu, std = _normalise(seq_tr, seq_te)
    assert norm_tr.shape == seq_tr.shape
    assert norm_te.shape == seq_te.shape
    assert mu.shape  == (5,)
    assert std.shape == (5,)


def test_normalise_no_leakage():
    """Test normalisation must use train stats, not test stats."""
    rng    = np.random.default_rng(2)
    seq_tr = rng.standard_normal((40, 12, 5)).astype(np.float32)
    # Test set has very different scale — if test stats leaked, train norm would differ
    seq_te = rng.standard_normal((15, 12, 5)).astype(np.float32) * 100
    norm_tr_a, _, mu_a, _ = _normalise(seq_tr, seq_te)
    norm_tr_b, _, mu_b, _ = _normalise(seq_tr, seq_tr)  # same train, different test
    assert np.allclose(mu_a, mu_b, atol=1e-5), \
        "Normalisation stats changed with different test set — leakage detected"


# ── Training + prediction tests ───────────────────────────────────────────────

def test_train_and_predict_shape():
    rng   = np.random.default_rng(3)
    X_seq = rng.standard_normal((30, 12, 5)).astype(np.float32)
    y     = rng.integers(0, 2, 30)
    model = _train_gru(X_seq, y, n_epochs=2, batch_size=16)
    probs = _predict_gru(model, X_seq)
    assert probs.shape == (30,)


def test_train_and_predict_range():
    rng   = np.random.default_rng(4)
    X_seq = rng.standard_normal((30, 12, 5)).astype(np.float32)
    y     = rng.integers(0, 2, 30)
    model = _train_gru(X_seq, y, n_epochs=2, batch_size=16)
    probs = _predict_gru(model, X_seq)
    assert probs.min() >= 0.0 - 1e-6
    assert probs.max() <= 1.0 + 1e-6


def test_gru_overfit_small_dataset():
    """
    GRU must be able to overfit a tiny perfectly separable dataset.
    If it can't, the architecture or training loop is broken.
    """
    torch.manual_seed(42)
    n      = 20
    X_pond = np.ones( (n//2, 12, 5), dtype=np.float32) *  1.0
    X_non  = np.ones( (n//2, 12, 5), dtype=np.float32) * -1.0
    X_seq  = np.concatenate([X_pond, X_non])
    y      = np.array([1]*(n//2) + [0]*(n//2))

    model = _train_gru(X_seq, y, n_epochs=200, batch_size=n, hidden_size=16)
    probs = _predict_gru(model, X_seq)
    preds = (probs >= 0.5).astype(int)
    accuracy = (preds == y).mean()
    assert accuracy >= 0.9, \
        f"GRU failed to overfit separable data: accuracy={accuracy:.2f}"