import json
import os
import sys
from typing import Dict

import numpy as np

# Ensure `<repo_root>/src` is on sys.path so `biomarker_driven_model` can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from biomarker_driven_model import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_logistic_regression(X: np.ndarray, y: np.ndarray) -> Dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    mean = X.mean(axis=0) if len(X) else np.zeros(X.shape[1], dtype=np.float64)
    scale = X.std(axis=0) if len(X) else np.ones(X.shape[1], dtype=np.float64)
    scale[scale == 0] = 1.0
    Xs = (X - mean) / scale

    n_samples, n_features = Xs.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        bias = 8.0 if positives > 0 else -8.0
        return {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
            "bias": float(bias),
            "constant_prediction": int(positives > 0),
        }

    sample_weights = np.where(
        y > 0.5,
        n_samples / (2.0 * positives),
        n_samples / (2.0 * negatives),
    )

    learning_rate = 0.1
    l2 = 1e-3
    steps = 4000

    for _ in range(steps):
        logits = Xs @ weights + bias
        probs = _sigmoid(logits)
        error = (probs - y) * sample_weights

        grad_w = (Xs.T @ error) / n_samples + l2 * weights
        grad_b = float(np.sum(error) / n_samples)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "constant_prediction": None,
    }


def train_fold(fold: int) -> None:
    fold = int(fold)
    path = os.path.join(config.features_dir, f"fold_{fold}.npz")
    data = np.load(path, allow_pickle=True)

    X = data["X_train"]
    y = data["y_train"].astype(int)

    model = _fit_logistic_regression(X, y)
    model.update(
        {
            "fold": fold,
            "n_train_videos": int(len(y)),
            "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
            "biomarkers": list(config.biomarkers),
            "feature_schema": ["presence_rate", "mean_area", "mean_count_per_frame"],
            "model": "numpy_logistic_regression",
        }
    )

    os.makedirs(config.models_dir, exist_ok=True)
    model_path = os.path.join(config.models_dir, f"model_fold_{fold}.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Saved fold {fold} model to {model_path}")


def main() -> None:
    for fold in range(int(config.kfolds)):
        train_fold(fold)


if __name__ == "__main__":
    main()
