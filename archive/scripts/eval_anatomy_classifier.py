#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple
import numpy as np

from _paths import add_archive_src_to_path

add_archive_src_to_path()

from anatomy_tracking import config
from anatomy_tracking.io import ensure_dir, read_csv, read_json, write_csv, write_json
from anatomy_tracking.metrics import binary_metrics, summarize_folds
from anatomy_tracking.modeling import load_model, predict_numpy_logistic_regression

#function for eval the anotomy classifier on existing folds.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default=os.path.join(config.FEATURES_DIR, "anatomy_features.csv"))
    parser.add_argument("--splits", default=config.DEFAULT_SPLITS_JSON)
    parser.add_argument("--models-dir", default=os.path.join(config.CLASSIFIER_RESULTS_DIR, "models"))
    parser.add_argument("--reports-dir", default=os.path.join(config.CLASSIFIER_RESULTS_DIR, "reports"))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    return parser.parse_args()

#feature matrix for anatomy classifier
def feature_matrix(rows: List[Dict], feature_names: List[str]) -> np.ndarray:
    return np.asarray([[float(row.get(name) or 0.0) for name in feature_names] for row in rows], dtype=np.float64)

#split rows for eval, ensuring all video_ids are present in the features table.
def split_rows(video_ids: List[str], rows_by_video: Dict[str, Dict], fold: int, split_name: str) -> List[Dict]:
    missing = [str(video_id) for video_id in video_ids if str(video_id) not in rows_by_video]
    if missing:
        raise KeyError(f"Fold {fold} {split_name} videos missing from anatomy feature table: {missing}")
    return [rows_by_video[str(video_id)] for video_id in video_ids]

#Evaluate a single fold of the anatomy classifier and save results and metrics.
def eval_fold(fold: int, rows_by_video: Dict[str, Dict], splits: Dict, models_dir: str, reports_dir: str) -> Tuple[List[Dict], Dict]:
    model = load_model(os.path.join(models_dir, f"anatomy_classifier_fold_{fold}.json"))
    feature_names = list(model["feature_names"])
    # Validation rows come from the shared split file, not a new anatomy-specific split.
    val_rows = split_rows(splits[f"fold_{fold}"]["val"], rows_by_video, fold, "val")
    X_val = feature_matrix(val_rows, feature_names)
    y_true = np.asarray([int(row["true_label"]) for row in val_rows], dtype=int)
    probs = predict_numpy_logistic_regression(model, X_val)
    preds = (probs >= 0.5).astype(int)

    result_rows = []
    for row, prob, pred in zip(val_rows, probs.tolist(), preds.tolist()):
        result_rows.append(
            {
                "fold": int(fold),
                "video_id": row["video_id"],
                "true_label": int(row["true_label"]),
                "predicted_label": int(pred),
                "predicted_prob": float(prob),
                "model": "anatomy_classifier",
            }
        )

    metrics = binary_metrics(y_true, probs, fold=fold)
    write_csv(os.path.join(reports_dir, f"fold_{fold}_results.csv"), result_rows)
    write_json(os.path.join(reports_dir, f"fold_{fold}_metrics.json"), metrics)
    return result_rows, metrics

#Evaluate all specified folds and save combined results and metrics.
def main() -> None:
    args = parse_args()
    ensure_dir(args.reports_dir)
    rows = read_csv(args.features_csv)
    rows_by_video = {str(row["video_id"]): row for row in rows}
    splits = read_json(args.splits)
    fold_ids = args.folds if args.folds is not None else sorted(int(key.split("_")[-1]) for key in splits)

    all_rows: List[Dict] = []
    all_metrics: List[Dict] = []
    for fold in fold_ids:
        rows_fold, metrics = eval_fold(fold, rows_by_video, splits, args.models_dir, args.reports_dir)
        all_rows.extend(rows_fold)
        all_metrics.append(metrics)
        print(f"Fold {fold}: accuracy={metrics['accuracy']:.3f} f1={metrics['f1_score']:.3f}")

    write_csv(os.path.join(args.reports_dir, "all_folds_results.csv"), all_rows)
    write_csv(os.path.join(args.reports_dir, "all_folds_metrics_by_folds.csv"), all_metrics)
    write_json(
        os.path.join(args.reports_dir, "all_folds_metrics.json"),
        {"folds": all_metrics, "summary": summarize_folds(all_metrics)},
    )

if __name__ == "__main__":
    main()
