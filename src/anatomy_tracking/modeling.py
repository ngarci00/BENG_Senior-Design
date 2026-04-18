import json
from typing import Dict
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

#Fit a logistic regression model using numpy by performing gradient descent optimization on the weights and bias, with feature scaling and handling of class imbalance through sample weighting.
def fit_numpy_logistic_regression(X: np.ndarray, y: np.ndarray, steps: int = 4000) -> Dict:
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
            "model": "numpy_logistic_regression",
        }

    sample_weights = np.where(
        y > 0.5,
        n_samples / (2.0 * positives),
        n_samples / (2.0 * negatives),
    )

    learning_rate = 0.1 #model's learning rate for gradient descent optimization
    l2 = 1e-3
    for _ in range(int(steps)):
        logits = Xs @ weights + bias
        probs = sigmoid(logits)
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
        "model": "numpy_logistic_regression",
    }
#Predict probabilities using a fitted logistic regression model, applying the same feature scaling and linear transformation as during training, followed by the sigmoid function to convert logits to probabilities.
def predict_numpy_logistic_regression(model: Dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if model.get("constant_prediction") is not None:
        return np.full((X.shape[0],), float(model["constant_prediction"]), dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    scale[scale == 0] = 1.0
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])
    return sigmoid(((X - mean) / scale) @ weights + bias)

def save_model(path: str, model: Dict) -> None:
    with open(path, "w") as f:
        json.dump(model, f, indent=2)

def load_model(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

