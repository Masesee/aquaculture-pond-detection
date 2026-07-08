"""
Single source of truth for loading and saving ensemble blend weights.
Never hardcode weight values in code — always load from the artifact
so production and tests stay identical by construction.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
WEIGHTS_PATH = MODELS_DIR / "blend_weights.json"


def load_blend_weights() -> tuple[float, float, float]:
    """
    Returns (w_lgbm, w_xgb, w_cb) as persisted in blend_weights.json.
    If the file does not exist, defaults to equal weights (1/3, 1/3, 1/3)
    and saves them to initialize the file.
    """
    if not WEIGHTS_PATH.exists():
        weights = (1/3, 1/3, 1/3)
        save_blend_weights(weights)
        return weights

    with open(WEIGHTS_PATH) as f:
        data = json.load(f)
    return (data["w_lgbm"], data["w_xgb"], data["w_cb"])


def save_blend_weights(weights: tuple[float, float, float]) -> None:
    """
    Saves (w_lgbm, w_xgb, w_cb) to blend_weights.json.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "w_lgbm": weights[0],
        "w_xgb": weights[1],
        "w_cb": weights[2],
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(data, f, indent=4)
