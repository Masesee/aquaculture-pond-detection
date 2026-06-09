"""
Evaluation metrics aligned with the Zindi scoring formula.
All functions are pure — no side effects, no file I/O.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def combined_score(f1: float, auc: float) -> float:
    """Zindi metric: 0.6 * F1 + 0.4 * AUC."""
    return 0.6 * f1 + 0.4 * auc


def evaluate(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Full evaluation at a fixed threshold.

    Returns dict with keys: f1, auc, score, n_predicted_positive,
    predicted_positive_rate.
    """
    preds = (probs >= threshold).astype(int)
    f1    = f1_score(labels, preds)
    auc   = roc_auc_score(labels, probs)

    return {
        "f1":                     round(f1,    6),
        "auc":                    round(auc,   6),
        "score":                  round(combined_score(f1, auc), 6),
        "n_predicted_positive":   int(preds.sum()),
        "predicted_positive_rate": round(float(preds.mean()), 4),
    }