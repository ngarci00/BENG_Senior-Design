#!/usr/bin/env python3
"""Train/evaluate the hybrid SVM and Mask R-CNN anatomy ensemble."""
import argparse
import os
import sys
from typing import Dict, List, Sequence

from _paths import PROJECT_ROOT, add_archive_src_to_path, archive_path, default_hybrid_reports_dir, project_path

add_archive_src_to_path()

from anatomy_tracking import config as tracking_config
from anatomy_tracking.io import DETECTION_FIELDNAMES, ensure_dir, read_csv, write_csv, write_json
from anatomy_tracking.metrics import summarize_folds


def repo_path(*parts: str) -> str:
    return project_path(*parts)


def script_path(name: str) -> str:
    return archive_path("scripts", name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=tracking_config.DEFAULT_INDEX_JSON)
    parser.add_argument("--splits", default=tracking_config.DEFAULT_SPLITS_JSON)
    parser.add_argument("--output-root", default=archive_path("outputs", "detector_hybrid_ensemble"))
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocess steps")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--skip-hybrid", action="store_true", help="Use existing hybrid reports instead of training SVM")
    parser.add_argument(
        "--hybrid-reports-dir",
        default=default_hybrid_reports_dir(),
        help="Hybrid SVM report directory used for late fusion",
    )
    parser.add_argument("--hybrid-weight", type=float, default=0.5)

    parser.add_argument("--skip-detector-training", action="store_true")
    parser.add_argument("--skip-detector-prediction", action="store_true")
    parser.add_argument("--detector-epochs", type=int, default=10)
    parser.add_argument("--detector-batch-size", type=int, default=1)
    parser.add_argument("--detector-device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--detector-min-size", type=int, default=320)
    parser.add_argument("--detector-max-size", type=int, default=512)
    parser.add_argument("--detector-frame-stride", type=int, default=1)
    parser.add_argument("--detector-max-frames-per-video", type=int, default=None)
    parser.add_argument("--detector-val-max-frames-per-video", type=int, default=32)
    parser.add_argument("--detector-max-train-samples", type=int, default=16)
    parser.add_argument("--detector-max-val-samples", type=int, default=None)
    parser.add_argument("--detector-no-pretrained", action="store_true")
    parser.add_argument("--detector-allow-random-fallback", action="store_true")

    parser.add_argument("--predict-frame-source", default="annotated", choices=["annotated", "all"])
    parser.add_argument("--predict-batch-size", type=int, default=2)
    parser.add_argument("--predict-frame-stride", type=int, default=1)
    parser.add_argument("--predict-progress-every", type=int, default=100)
    parser.add_argument("--predict-score-threshold", type=float, default=0.5)
    parser.add_argument("--predict-mask-threshold", type=float, default=0.5)
    parser.add_argument("--predict-max-detections-per-frame", type=int, default=20)

    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--skip-anatomy-classifier", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: Sequence[str], dry_run: bool = False) -> None:
    print(f"\n[{name}]", flush=True)
    print(" ".join(command), flush=True)
    if dry_run:
        return
    import subprocess

    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def add_optional_int(command: List[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def detector_train_command(args: argparse.Namespace, fold: int, detector_dir: str) -> List[str]:
    command = [
        args.python,
        script_path("train_anatomy_maskrcnn.py"),
        "--index",
        args.index,
        "--splits",
        args.splits,
        "--fold",
        str(fold),
        "--output-dir",
        detector_dir,
        "--epochs",
        str(args.detector_epochs),
        "--batch-size",
        str(args.detector_batch_size),
        "--device",
        args.detector_device,
        "--min-size",
        str(args.detector_min_size),
        "--max-size",
        str(args.detector_max_size),
        "--frame-stride",
        str(args.detector_frame_stride),
        "--val-max-frames-per-video",
        str(args.detector_val_max_frames_per_video),
    ]
    add_optional_int(command, "--max-frames-per-video", args.detector_max_frames_per_video)
    add_optional_int(command, "--max-train-samples", args.detector_max_train_samples)
    add_optional_int(command, "--max-val-samples", args.detector_max_val_samples)
    if args.detector_no_pretrained:
        command.append("--no-pretrained")
    if args.detector_allow_random_fallback:
        command.append("--allow-random-fallback")
    return command


def detector_predict_command(
    args: argparse.Namespace,
    checkpoint: str,
    fold: int,
    split: str,
    output_dir: str,
) -> List[str]:
    return [
        args.python,
        script_path("predict_anatomy_maskrcnn.py"),
        "--checkpoint",
        checkpoint,
        "--index",
        args.index,
        "--splits",
        args.splits,
        "--fold",
        str(fold),
        "--split",
        split,
        "--batch-size",
        str(args.predict_batch_size),
        "--frame-source",
        args.predict_frame_source,
        "--frame-stride",
        str(args.predict_frame_stride),
        "--progress-every",
        str(args.predict_progress_every),
        "--score-threshold",
        str(args.predict_score_threshold),
        "--mask-threshold",
        str(args.predict_mask_threshold),
        "--max-detections-per-frame",
        str(args.predict_max_detections_per_frame),
        "--device",
        args.detector_device,
        "--output-dir",
        output_dir,
    ]


def combine_detection_csvs(input_paths: Sequence[str], output_path: str, dry_run: bool = False) -> None:
    print(f"\n[Combine detector detections]\n{output_path}", flush=True)
    if dry_run:
        return
    rows: List[Dict] = []
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows.extend(read_csv(path))
    write_csv(output_path, rows, DETECTION_FIELDNAMES)
    print(f"Wrote {len(rows)} combined detections to {output_path}", flush=True)


def aggregate_anatomy_reports(reports_dir: str, folds: Sequence[int], dry_run: bool = False) -> None:
    print(f"\n[Aggregate detector-anatomy reports]\n{reports_dir}", flush=True)
    if dry_run:
        return
    all_rows: List[Dict] = []
    all_metrics: List[Dict] = []
    for fold in folds:
        fold_results = os.path.join(reports_dir, f"fold_{fold}_results.csv")
        fold_metrics = os.path.join(reports_dir, f"fold_{fold}_metrics.json")
        if not os.path.exists(fold_results):
            raise FileNotFoundError(fold_results)
        if not os.path.exists(fold_metrics):
            raise FileNotFoundError(fold_metrics)
        all_rows.extend(read_csv(fold_results))
        import json

        with open(fold_metrics, "r") as f:
            all_metrics.append(json.load(f))
    write_csv(os.path.join(reports_dir, "all_folds_results.csv"), all_rows)
    write_csv(os.path.join(reports_dir, "all_folds_metrics_by_folds.csv"), all_metrics)
    write_json(
        os.path.join(reports_dir, "all_folds_metrics.json"),
        {"folds": all_metrics, "summary": summarize_folds(all_metrics)},
    )


def main() -> None:
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    folds = [int(fold) for fold in args.folds]

    detector_dir = os.path.join(output_root, "maskrcnn_detector")
    predictions_dir = os.path.join(output_root, "detector_predictions")
    tracks_root = os.path.join(output_root, "anatomy_tracks")
    features_root = os.path.join(output_root, "anatomy_features")
    anatomy_models_dir = os.path.join(output_root, "anatomy_classifier_results", "models")
    anatomy_reports_dir = os.path.join(output_root, "anatomy_classifier_results", "reports")
    ensemble_dir = os.path.join(output_root, "ensemble_results")
    comparison_dir = os.path.join(output_root, "model_comparison")

    if not args.skip_hybrid:
        run_step(
            "Train/evaluate hybrid SVM branch",
            [args.python, project_path("src", "run_HYBRID", "run.py")],
            args.dry_run,
        )

    for fold in folds:
        checkpoint = os.path.join(detector_dir, f"fold_{fold}", "maskrcnn_best.pt")
        if not args.skip_detector_training:
            run_step(
                f"Train Mask R-CNN detector fold {fold}",
                detector_train_command(args, fold, detector_dir),
                args.dry_run,
            )

        train_pred_dir = os.path.join(predictions_dir, f"fold_{fold}_train")
        val_pred_dir = os.path.join(predictions_dir, f"fold_{fold}_val")
        if not args.skip_detector_prediction:
            run_step(
                f"Predict detector train split fold {fold}",
                detector_predict_command(args, checkpoint, fold, "train", train_pred_dir),
                args.dry_run,
            )
            run_step(
                f"Predict detector val split fold {fold}",
                detector_predict_command(args, checkpoint, fold, "val", val_pred_dir),
                args.dry_run,
            )

        combined_dir = os.path.join(predictions_dir, f"fold_{fold}_train_val")
        combined_csv = os.path.join(combined_dir, "detections.csv")
        combine_detection_csvs(
            [
                os.path.join(train_pred_dir, "detections.csv"),
                os.path.join(val_pred_dir, "detections.csv"),
            ],
            combined_csv,
            dry_run=args.dry_run,
        )

        tracks_dir = os.path.join(tracks_root, f"fold_{fold}")
        features_dir = os.path.join(features_root, f"fold_{fold}")
        if not args.skip_tracking:
            run_step(
                f"Track predicted anatomy fold {fold}",
                [
                    args.python,
                    script_path("track_anatomy.py"),
                    "--detections-csv",
                    combined_csv,
                    "--output-dir",
                    tracks_dir,
                ],
                args.dry_run,
            )
            run_step(
                f"Extract detector-anatomy features fold {fold}",
                [
                    args.python,
                    script_path("extract_anatomy_features.py"),
                    "--tracks-csv",
                    os.path.join(tracks_dir, "tracks.csv"),
                    "--index",
                    args.index,
                    "--output-dir",
                    features_dir,
                ],
                args.dry_run,
            )

        if not args.skip_anatomy_classifier:
            features_csv = os.path.join(features_dir, "anatomy_features.csv")
            run_step(
                f"Train detector-anatomy classifier fold {fold}",
                [
                    args.python,
                    script_path("train_anatomy_classifier.py"),
                    "--features-csv",
                    features_csv,
                    "--splits",
                    args.splits,
                    "--output-dir",
                    anatomy_models_dir,
                    "--folds",
                    str(fold),
                ],
                args.dry_run,
            )
            run_step(
                f"Evaluate detector-anatomy classifier fold {fold}",
                [
                    args.python,
                    script_path("eval_anatomy_classifier.py"),
                    "--features-csv",
                    features_csv,
                    "--splits",
                    args.splits,
                    "--models-dir",
                    anatomy_models_dir,
                    "--reports-dir",
                    anatomy_reports_dir,
                    "--folds",
                    str(fold),
                ],
                args.dry_run,
            )

    if not args.skip_anatomy_classifier:
        aggregate_anatomy_reports(anatomy_reports_dir, folds, dry_run=args.dry_run)

    if not args.skip_ensemble:
        run_step(
            "Fuse hybrid and detector-anatomy predictions",
            [
                args.python,
                script_path("ensemble_predictions.py"),
                "--hybrid-reports-dir",
                args.hybrid_reports_dir,
                "--anatomy-reports-dir",
                anatomy_reports_dir,
                "--output-dir",
                ensemble_dir,
                "--hybrid-weight",
                str(args.hybrid_weight),
                "--folds",
                *[str(fold) for fold in folds],
            ],
            args.dry_run,
        )
        run_step(
            "Compare hybrid, detector-anatomy, and ensemble",
            [
                args.python,
                script_path("compare_models.py"),
                "--hybrid-results",
                os.path.join(args.hybrid_reports_dir, "all_folds_results.csv"),
                "--anatomy-results",
                os.path.join(anatomy_reports_dir, "all_folds_results.csv"),
                "--ensemble-results",
                os.path.join(ensemble_dir, "all_folds_results.csv"),
                "--output-dir",
                comparison_dir,
            ],
            args.dry_run,
        )

    if not args.dry_run:
        write_json(
            os.path.join(output_root, "run_summary.json"),
            {
                "folds": folds,
                "index": os.path.abspath(args.index),
                "splits": os.path.abspath(args.splits),
                "hybrid_reports_dir": os.path.abspath(args.hybrid_reports_dir),
                "anatomy_reports_dir": os.path.abspath(anatomy_reports_dir),
                "ensemble_dir": os.path.abspath(ensemble_dir),
                "comparison_dir": os.path.abspath(comparison_dir),
                "predict_frame_source": args.predict_frame_source,
            },
        )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
