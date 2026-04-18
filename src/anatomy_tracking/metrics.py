import math
from typing import Dict, List
import numpy as np

#Compute confusion matrix counts (TP, TN, FP, FN) for binary classification given true labels and predicted labels.
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }

#Compute binary classification metrics such as acc, f1 score, precision, recall etc, given true labels and predicted probabilities, with optional thresholding and fold information for reporting.
def binary_metrics(y_true, y_prob, threshold: float = 0.5, fold=None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)

    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    specificity = tn / float(tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    bal_accuracy = 0.5 * (recall + specificity)

    metrics = {
        "n_videos": int(len(y_true)),
        "accuracy": accuracy,
        "bal_accuracy": bal_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    if fold is not None:
        metrics = {"fold": int(fold), **metrics}
    metrics["roc_auc"] = roc_auc_score_safe(y_true, y_prob)
    return metrics

#Compute ROC AUC score safely by handling cases with only one class present, which would otherwise cause errors in standard implementations.
def roc_auc_score_safe(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)

    positives = y_true == 1
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(~positives))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

#Summarize metrics across multiple folds by computing mean and standard deviation for key metrics, which provides an overall assessment of model performance and variability across different data splits.
def summarize_folds(metrics: List[Dict]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    keys = [
        "accuracy",
        "bal_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "n_videos",
    ]
    for key in keys:
        vals = []
        for item in metrics:
            value = item.get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                vals.append(float(value))
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    return summary

