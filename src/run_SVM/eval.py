import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)


# Ensure `<repo_root>/src` is on sys.path so we can import run_SVM/config reliably
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from run_SVM import config  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return roc_auc


def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec, prec))

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AUC={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return pr_auc


def _add_mean_std(metrics: List[Dict]) -> Dict:
    """Compute mean/std summary across folds for common numeric keys."""
    if not metrics:
        return {}

    keys = [
        "accuracy",
        "bal_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "n_videos",
    ]

    summary: Dict[str, float] = {}
    for k in keys:
        vals = [m.get(k, None) for m in metrics]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            continue
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=0))

    return summary


def _infer_fold_svm(fold: int, report_dir: str) -> Tuple[List[Dict], Dict[str, float]]:
    """Return per-video rows and a metrics dict for one fold."""
    feats_path = os.path.join(config.features_dir, f"fold_{fold}.npz")
    model_path = os.path.join(config.models_dir, f"svm_fold_{fold}.joblib")

    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Missing features for fold {fold}: {feats_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing SVM model for fold {fold}: {model_path}")

    data = np.load(feats_path, allow_pickle=True)
    X_val = data["X_val"]
    y_val = data["y_val"].astype(int).reshape(-1)

    # Prefer stored video ids; otherwise synthesize stable ids
    if "vids_val" in data.files:
        try:
            vids_val = [str(v) for v in data["vids_val"]]
        except Exception:
            vids_val = [f"val_{i}" for i in range(len(y_val))]
    else:
        vids_val = [f"val_{i}" for i in range(len(y_val))]

    model = joblib.load(model_path)

    y_pred = model.predict(X_val).astype(int).reshape(-1)

    # Probabilities (PASS=1) if available; else decision_function -> sigmoid
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_val)[:, 1].astype(float)
        except Exception:
            y_prob = None

    if y_prob is None and hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_val).astype(float)
            y_prob = _sigmoid(scores)
        except Exception:
            y_prob = None

    # Per-video rows (CNN-style schema)
    rows: List[Dict] = []
    for vid, yt, yp in zip(vids_val, y_val.tolist(), y_pred.tolist()):
        pr = float(y_prob[rows.__len__()]) if y_prob is not None else float("nan")
        rows.append(
            {
                "fold": int(fold),
                "video_id": str(vid),
                "true_label": int(yt),
                "predicted_label": int(yp),
                "predicted_prob": pr,
                "prob_mean": pr,
                "n_frames": int(-1),
            }
        )

    # Metrics
    acc = float(accuracy_score(y_val, y_pred))
    bal_acc = float(balanced_accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, zero_division=0))
    rec = float(recall_score(y_val, y_pred, zero_division=0))
    f1 = float(f1_score(y_val, y_pred, zero_division=0))

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    metrics: Dict[str, float] = {
        "fold": int(fold),
        "n_videos": int(len(y_val)),
        "accuracy": acc,
        "bal_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }

    # Plots (CNN-style filenames)
    cm_path = os.path.join(report_dir, f"fold_{fold}_confusion_matrix.png")
    _plot_confusion_matrix(cm, cm_path, title=f"Fold {fold} Confusion Matrix")

    if y_prob is not None and len(np.unique(y_val)) == 2:
        roc_path = os.path.join(report_dir, f"fold_{fold}_roc.png")
        pr_path = os.path.join(report_dir, f"fold_{fold}_pr.png")
        metrics["roc_auc"] = float(_plot_roc(y_val, y_prob, roc_path, title=f"Fold {fold} ROC"))
        metrics["pr_auc"] = float(_plot_pr(y_val, y_prob, pr_path, title=f"Fold {fold} Precision-Recall"))

    # Write per-fold outputs
    _write_csv(os.path.join(report_dir, f"fold_{fold}_results.csv"), rows)
    _write_json(os.path.join(report_dir, f"fold_{fold}_metrics.json"), metrics)

    return rows, metrics


def eval_fold(fold: int, report_dir: str | None = None) -> Dict[str, float]:
    """Public API used by run.py.

    Evaluates one fold and writes the per-fold CNN-style artifacts into report_dir.
    Returns the metrics dict.
    """
    rd = report_dir if report_dir else config.reports_dir
    _ensure_dir(rd)
    _rows, metrics = _infer_fold_svm(int(fold), report_dir=rd)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Evaluate only one fold")
    parser.add_argument("--report_dir", type=str, default=None, help="Override report directory")
    args = parser.parse_args()

    report_dir = args.report_dir if args.report_dir else config.reports_dir
    _ensure_dir(report_dir)

    folds = [args.fold] if args.fold is not None else list(range(int(config.kfolds)))

    all_rows: List[Dict] = []
    all_metrics: List[Dict] = []

    for fold in folds:
        rows, metrics = _infer_fold_svm(int(fold), report_dir=report_dir)
        all_rows.extend(rows)
        all_metrics.append(metrics)
        print(metrics)

    # All-fold outputs (CNN-style names)
    _write_csv(os.path.join(report_dir, "all_folds_results.csv"), all_rows)
    _write_csv(os.path.join(report_dir, "all_folds_metrics_by_folds.csv"), all_metrics)

    summary = _add_mean_std(all_metrics)
    _write_csv(os.path.join(report_dir, "all_folds_summary.csv"), [summary] if summary else [])
    _write_json(os.path.join(report_dir, "all_folds_metrics.json"), {"by_fold": all_metrics, "summary": summary})

    print(f"[SVM] Wrote reports to: {report_dir}")


if __name__ == "__main__":
    main()