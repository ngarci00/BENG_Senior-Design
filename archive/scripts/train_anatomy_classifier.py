#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple
import numpy as np

from _paths import add_archive_src_to_path

add_archive_src_to_path()

from anatomy_tracking import config
from anatomy_tracking.io import ensure_dir, read_csv, read_json, write_json
from anatomy_tracking.modeling import fit_numpy_logistic_regression, save_model

#Simple logistic regression classifier trained on anatomy tracking features to predict true video labels, used as a standalone anatomy-only model and as a component of the hybrid ensemble.
NON_FEATURE_COLUMNS = {"video_id", "true_label"}

#function to parse command line arguments for the anatomy classifier training script, with defaults and help messages for each argument.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default=os.path.join(config.FEATURES_DIR, "anatomy_features.csv"))
    parser.add_argument("--splits", default=config.DEFAULT_SPLITS_JSON)
    parser.add_argument("--output-dir", default=os.path.join(config.CLASSIFIER_RESULTS_DIR, "models"))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    return parser.parse_args()

#Utility function to load the anatomy feature table from a CSV file, returning both the rows and the list of feature names (excluding non-feature metadata columns).
def load_feature_table(path: str) -> Tuple[List[Dict], List[str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    #All non-label metadata columns are excluded from the classifier matrix.
    feature_names = [
        key
        for key in rows[0].keys()
        if key not in NON_FEATURE_COLUMNS and key != ""
    ]
    return rows, feature_names

#Utility function to convert a list of feature rows into numpy arrays for training the logistic regression model, separating out the feature matrix X, label vector y, and video ID array vids.
def rows_to_matrix(rows: List[Dict], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray([[float(row.get(name) or 0.0) for name in feature_names] for row in rows], dtype=np.float64)
    y = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    vids = np.asarray([str(row["video_id"]) for row in rows], dtype=object)
    return X, y, vids

#Utility function to split the feature rows into training sets based on the provided video splits, ensuring that all videos in the specified fold and split are present in the feature table.
def split_rows(video_ids: List[str], rows_by_video: Dict[str, Dict], fold: int, split_name: str) -> List[Dict]:
    missing = [str(video_id) for video_id in video_ids if str(video_id) not in rows_by_video]
    if missing:
        raise KeyError(f"Fold {fold} {split_name} videos missing from anatomy feature table: {missing}")
    return [rows_by_video[str(video_id)] for video_id in video_ids]

#Main function to orchestrate the entire pipeline, running each step in sequence with appropriate arguments and handling output directories and file paths.
def main() -> None:
    args = parse_args()
    rows, feature_names = load_feature_table(args.features_csv)
    splits = read_json(args.splits)
    rows_by_video = {str(row["video_id"]): row for row in rows}
    fold_ids = args.folds if args.folds is not None else sorted(int(key.split("_")[-1]) for key in splits)

    ensure_dir(args.output_dir)
    for fold in fold_ids:
        split = splits[f"fold_{fold}"]
        # Reuse the same video folds as the hybrid model for direct comparison.
        train_rows = split_rows(split["train"], rows_by_video, fold, "train")
        X_train, y_train, _vids_train = rows_to_matrix(train_rows, feature_names)
        model = fit_numpy_logistic_regression(X_train, y_train)
        model.update(
            {
                "fold": int(fold),
                "feature_names": feature_names,
                "n_train_videos": int(len(y_train)),
                "features_csv": os.path.abspath(args.features_csv),
                "splits": os.path.abspath(args.splits),
            }
        )
        output_path = os.path.join(args.output_dir, f"anatomy_classifier_fold_{fold}.json")
        save_model(output_path, model)
        print(f"Saved anatomy classifier fold {fold} model to {output_path}")

    write_json(
        os.path.join(args.output_dir, "training_summary.json"),
        {
            "features_csv": os.path.abspath(args.features_csv),
            "splits": os.path.abspath(args.splits),
            "feature_names": feature_names,
            "folds": fold_ids,
        },
    )


if __name__ == "__main__":
    main()
