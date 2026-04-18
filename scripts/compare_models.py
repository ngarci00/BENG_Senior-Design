#!/usr/bin/env python3
"""Compare hybrid, anatomy-only, and ensemble video-level results."""
import argparse
import os
import sys
from typing import Dict, List
import numpy as np
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
# Use repo-local helpers without requiring an editable install.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from anatomy_tracking import config
from anatomy_tracking.io import ensure_dir, read_csv, write_csv, write_json
from anatomy_tracking.metrics import confusion_counts, roc_auc_score_safe

#The default paths are set to the expected output locations of the respective branches, but can be overridden to compare other result files.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-results",
        default=os.path.join(REPO_ROOT, "runs", "run_SVM", "res_eval", "reports_50Poly_224x224", "all_folds_results.csv"),
    )
    parser.add_argument(
        "--anatomy-results",
        default=os.path.join(config.CLASSIFIER_RESULTS_DIR, "reports", "all_folds_results.csv"),
    )
    parser.add_argument(
        "--ensemble-results",
        default=os.path.join(config.ENSEMBLE_RESULTS_DIR, "all_folds_results.csv"),
    )
    parser.add_argument("--output-dir", default=os.path.join(config.DEFAULT_OUTPUT_DIR, "model_comparison"))
    return parser.parse_args()

#extracting probabilities from the row, prioritizing keys that may be present in different result files, and falling back to predicted_label if no probability keys are found.
def probability(row: Dict) -> float:
    # Preserve probabilities for ROC-AUC even though labels are scored from predicted_label.
    for key in ["predicted_prob", "prob_mean", "logit_mean"]:
        if key in row and row[key] != "":
            return float(row[key])
    return float(row["predicted_label"])

#Load rows from a CSV file into a dictionary keyed by "fold:video_id" for easy comparison across models, and include the model name in the row data.
def load_named_rows(path: str, name: str) -> Dict[str, Dict]:
    rows = {}
    for row in read_csv(path):
        key = f"{row.get('fold', '')}:{row['video_id']}"
        rows[key] = {
            "model": name,
            "fold": int(row.get("fold", 0)),
            "video_id": row["video_id"],
            "true_label": int(row["true_label"]),
            "predicted_label": int(row["predicted_label"]),
            "predicted_prob": probability(row),
        }
    return rows

#Compute various classification metrics for a given model based on the true labels and predicted probabilities/labels across a set of rows corresponding to fold/video_id combinations.
def metric_row(name: str, rows: List[Dict]) -> Dict:
    y_true = np.asarray([row["true_label"] for row in rows], dtype=int)
    probs = np.asarray([row["predicted_prob"] for row in rows], dtype=float)
    # Use saved predicted_label; SVM predict_proba can disagree with SVC.predict near 0.5.
    preds = np.asarray([row["predicted_label"] for row in rows], dtype=int)
    counts = confusion_counts(y_true, preds)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    specificity = tn / float(tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    metrics = {
        "n_videos": int(len(y_true)),
        "accuracy": float(np.mean(preds == y_true)) if len(y_true) else 0.0,
        "bal_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "roc_auc": roc_auc_score_safe(y_true, probs),
    }
    return {"model": name, **metrics}

#Main function:
def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    models = {
        "hybrid": load_named_rows(args.hybrid_results, "hybrid"),
        "anatomy": load_named_rows(args.anatomy_results, "anatomy"),
        "ensemble": load_named_rows(args.ensemble_results, "ensemble"),
    }
    common = sorted(set.intersection(*(set(rows) for rows in models.values())))
    if not common:
        raise SystemExit("No common fold/video rows across requested result files.")

    # Metrics are computed on the shared fold/video rows across all branches.
    metrics = [metric_row(name, [rows[key] for key in common]) for name, rows in models.items()]
    write_csv(os.path.join(args.output_dir, "model_metrics_summary.csv"), metrics)

    comparison_rows = []
    failure_rows = []
    for key in common:
        per_model = {name: rows[key] for name, rows in models.items()}
        truth = per_model["hybrid"]["true_label"]
        row = {
            "fold": per_model["hybrid"]["fold"],
            "video_id": per_model["hybrid"]["video_id"],
            "true_label": truth,
        }
        for name, result in per_model.items():
            correct = int(result["predicted_label"] == truth)
            row[f"{name}_predicted_label"] = result["predicted_label"]
            row[f"{name}_predicted_prob"] = result["predicted_prob"]
            row[f"{name}_correct"] = correct
        comparison_rows.append(row)

        if row["hybrid_correct"] == 0 or row["anatomy_correct"] == 0 or row["ensemble_correct"] == 0:
            failure_rows.append(
                {
                    **row,
                    "ensemble_fixes_hybrid": int(row["hybrid_correct"] == 0 and row["ensemble_correct"] == 1),
                    "hybrid_wrong_anatomy_right": int(row["hybrid_correct"] == 0 and row["anatomy_correct"] == 1),
                    "anatomy_wrong_hybrid_right": int(row["anatomy_correct"] == 0 and row["hybrid_correct"] == 1),
                }
            )

    write_csv(os.path.join(args.output_dir, "per_video_model_comparison.csv"), comparison_rows)
    write_csv(os.path.join(args.output_dir, "failure_case_comparison.csv"), failure_rows)
    write_json(
        os.path.join(args.output_dir, "comparison_summary.json"),
        {
            "n_common_rows": len(common),
            "metrics": metrics,
            "n_failure_rows": len(failure_rows),
            "n_ensemble_fixes_hybrid": int(sum(row.get("ensemble_fixes_hybrid", 0) for row in failure_rows)),
            "n_hybrid_wrong_anatomy_right": int(sum(row.get("hybrid_wrong_anatomy_right", 0) for row in failure_rows)),
            "n_anatomy_wrong_hybrid_right": int(sum(row.get("anatomy_wrong_hybrid_right", 0) for row in failure_rows)),
        },
    )
    print(f"Wrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
