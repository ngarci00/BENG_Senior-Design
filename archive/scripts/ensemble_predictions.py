#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple
import numpy as np

from _paths import add_archive_src_to_path, project_path

add_archive_src_to_path()

from anatomy_tracking import config
from anatomy_tracking.io import ensure_dir, read_csv, write_csv, write_json
from anatomy_tracking.metrics import binary_metrics, summarize_folds

#Ensemble video-level predictions from the 50-video hybrid model and anatomy classifier by late averaging of probabilities, then compute metrics.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-reports-dir",
        default=project_path("runs", "run_SVM", "res_eval", "reports_50Poly_224x224"),
        help="Directory containing 50-video hybrid fold_*_results.csv files",
    )
    parser.add_argument(
        "--anatomy-reports-dir",
        default=os.path.join(config.CLASSIFIER_RESULTS_DIR, "reports"),
        help="Directory containing anatomy classifier fold_*_results.csv files",
    )
    parser.add_argument("--output-dir", default=config.ENSEMBLE_RESULTS_DIR)
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--hybrid-weight", type=float, default=0.5)
    return parser.parse_args()

#Load video-level prediction rows for a given fold from the specified report directory.
def load_fold_rows(report_dir: str, fold: int) -> Dict[str, Dict]:
    path = os.path.join(report_dir, f"fold_{fold}_results.csv")
    return {str(row["video_id"]): row for row in read_csv(path)}

#Extract the predicted probability for the positive class from a report row, handling different column names.
def probability(row: Dict) -> float:
    #SVM and anatomy reports use slightly different probability column names.
    for key in ["predicted_prob", "prob_mean", "logit_mean"]:
        if key in row and row[key] != "":
            return float(row[key])
    return float(row["predicted_label"])

#Fuse predictions for a single fold by late averaging of video-level probabilities, then compute metrics.
def fuse_fold(args: argparse.Namespace, fold: int) -> Tuple[List[Dict], Dict]:
    hybrid = load_fold_rows(args.hybrid_reports_dir, fold)
    anatomy = load_fold_rows(args.anatomy_reports_dir, fold)
    #Intersect by video id so missing annotations do not silently misalign rows.
    common_ids = sorted(set(hybrid).intersection(anatomy))
    if not common_ids:
        raise ValueError(f"No overlapping video IDs for fold {fold}")

    rows = []
    y_true = []
    probs = []
    w_hybrid = float(args.hybrid_weight)
    w_anatomy = 1.0 - w_hybrid
    for video_id in common_ids:
        h = hybrid[video_id]
        a = anatomy[video_id]
        true_label = int(h["true_label"])
        if true_label != int(a["true_label"]):
            raise ValueError(f"Label mismatch for video {video_id} in fold {fold}")
        hybrid_prob = probability(h)
        anatomy_prob = probability(a)
        # Late fusion baseline: weighted average of video-level PASS probabilities.
        ensemble_prob = w_hybrid * hybrid_prob + w_anatomy * anatomy_prob
        pred = int(ensemble_prob >= 0.5)
        rows.append(
            {
                "fold": int(fold),
                "video_id": video_id,
                "true_label": true_label,
                "hybrid_prob": float(hybrid_prob),
                "anatomy_prob": float(anatomy_prob),
                "predicted_prob": float(ensemble_prob),
                "predicted_label": pred,
                "hybrid_weight": w_hybrid,
                "anatomy_weight": w_anatomy,
            }
        )
        y_true.append(true_label)
        probs.append(ensemble_prob)
    return rows, binary_metrics(np.asarray(y_true), np.asarray(probs), fold=fold)

#Main entry point to run the ensemble fusion across specified folds and save results.
def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    all_rows: List[Dict] = []
    all_metrics: List[Dict] = []

    for fold in args.folds:
        rows, metrics = fuse_fold(args, fold)
        all_rows.extend(rows)
        all_metrics.append(metrics)
        write_csv(os.path.join(args.output_dir, f"fold_{fold}_results.csv"), rows)
        write_json(os.path.join(args.output_dir, f"fold_{fold}_metrics.json"), metrics)
        print(f"Fold {fold}: accuracy={metrics['accuracy']:.3f} f1={metrics['f1_score']:.3f}")

    write_csv(os.path.join(args.output_dir, "all_folds_results.csv"), all_rows)
    write_csv(os.path.join(args.output_dir, "all_folds_metrics_by_folds.csv"), all_metrics)
    write_json(
        os.path.join(args.output_dir, "all_folds_metrics.json"),
        {"folds": all_metrics, "summary": summarize_folds(all_metrics)},
    )

if __name__ == "__main__":
    main()
