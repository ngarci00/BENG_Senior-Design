import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Ensure `<repo_root>/src` is on sys.path so `biomarker_driven_model` can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from biomarker_driven_model import config


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float(np.mean(y_pred[pos_mask] == 1)) if np.any(pos_mask) else 0.0
    tnr = float(np.mean(y_pred[neg_mask] == 0)) if np.any(neg_mask) else 0.0
    return 0.5 * (tpr + tnr)


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tp + fp
    return float(tp / denom) if denom else 0.0


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = tp + fn
    return float(tp / denom) if denom else 0.0


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec = _precision(y_true, y_pred)
    rec = _recall(y_true, y_pred)
    denom = prec + rec
    return float(2.0 * prec * rec / denom) if denom else 0.0


def _load_model(fold: int) -> Dict:
    with open(os.path.join(config.models_dir, f"model_fold_{fold}.json"), "r") as f:
        return json.load(f)


def _predict_proba(model: Dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    scale[scale == 0] = 1.0
    Xs = (X - mean) / scale

    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])
    logits = Xs @ weights + bias
    return _sigmoid(logits)


def eval_fold(fold: int) -> Tuple[List[Dict], Dict[str, float]]:
    fold = int(fold)
    data = np.load(os.path.join(config.features_dir, f"fold_{fold}.npz"), allow_pickle=True)

    X = data["X_val"]
    y = data["y_val"].astype(int).reshape(-1)
    vids = [str(v) for v in data["vids_val"].tolist()]

    model = _load_model(fold)
    probs = _predict_proba(model, X)
    preds = (probs >= 0.5).astype(int)

    rows: List[Dict] = []
    for vid, yt, yp, prob in zip(vids, y.tolist(), preds.tolist(), probs.tolist()):
        rows.append(
            {
                "fold": fold,
                "video_id": vid,
                "true_label": int(yt),
                "predicted_label": int(yp),
                "predicted_prob": float(prob),
                "logit_mean": float(prob),
                "n_clips": 1,
            }
        )

    tp = int(np.sum((y == 1) & (preds == 1)))
    tn = int(np.sum((y == 0) & (preds == 0)))
    fp = int(np.sum((y == 0) & (preds == 1)))
    fn = int(np.sum((y == 1) & (preds == 0)))
    metrics: Dict[str, float] = {
        "fold": int(fold),
        "n_videos": int(len(y)),
        "accuracy": float(np.mean(preds == y)) if len(y) else 0.0,
        "bal_accuracy": _balanced_accuracy(y, preds),
        "precision": _precision(y, preds),
        "recall": _recall(y, preds),
        "f1_score": _f1(y, preds),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }

    _ensure_dir(config.reports_dir)
    _write_csv(os.path.join(config.reports_dir, f"fold_{fold}_results.csv"), rows)
    _write_json(os.path.join(config.reports_dir, f"fold_{fold}_metrics.json"), metrics)
    return rows, metrics


def write_aggregate_reports(all_rows: List[Dict], all_metrics: List[Dict]) -> None:
    _write_csv(os.path.join(config.reports_dir, "all_folds_results.csv"), all_rows)
    _write_csv(os.path.join(config.reports_dir, "all_folds_metrics_by_folds.csv"), all_metrics)
    _write_json(os.path.join(config.reports_dir, "all_folds_metrics.json"), all_metrics)


def main() -> None:
    all_rows: List[Dict] = []
    all_metrics: List[Dict] = []

    for fold in range(int(config.kfolds)):
        rows, metrics = eval_fold(fold)
        all_rows.extend(rows)
        all_metrics.append(metrics)

    write_aggregate_reports(all_rows, all_metrics)


if __name__ == "__main__":
    main()
