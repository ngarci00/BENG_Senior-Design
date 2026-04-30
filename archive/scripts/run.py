#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from typing import List, Optional, Sequence

from _paths import PROJECT_ROOT, archive_path, default_hybrid_reports_dir, project_path

SCRIPTS_DIR = archive_path("scripts")

#util function to contruct paths relative to the repo root, used for default argument values and script paths
def repo_path(*parts: str) -> str:
    return project_path(*parts)

#function to parse command line arguments for the main pipeline.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=repo_path("data", "videos", "json_utils", "index_poly.json"))
    parser.add_argument("--splits", default=repo_path("data", "videos", "json_utils", "splits_poly_50.json"))
    parser.add_argument(
        "--hybrid-reports-dir",
        default=default_hybrid_reports_dir(),
        help="Existing 50-video hybrid SVM report directory",
    )
    parser.add_argument("--output-root", default=archive_path("outputs"))
    parser.add_argument("--hybrid-weight", type=float, default=0.5)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--skip-audit", action="store_true", help="Skip annotation audit for faster reruns")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()

#Utility functions to run each step of the pipeline as a subprocess
def script_path(name: str) -> str:
    return os.path.join(SCRIPTS_DIR, name)

#Utility function to run a command as a subprocess, with optional dry-run mode to just print the command without executing it.
def run_step(name: str, command: Sequence[str], dry_run: bool = False) -> None:
    print(f"\n[{name}]", flush=True)
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

#Utility function to append fold arguments to commands that support running on specific folds, used for faster iteration during development and debugging.
def append_folds(command: List[str], folds: Optional[Sequence[int]]) -> List[str]:
    if folds:
        command.extend(["--folds", *[str(fold) for fold in folds]])
    return command

#Main function to orchestrate the entire pipeline, running each step in sequence with appropriate arguments and handling output directories and file paths.
def main() -> None:
    args = parse_args()
    output_root = os.path.abspath(args.output_root)

    audit_dir = os.path.join(output_root, "anatomy_label_audit")
    tracking_format_dir = os.path.join(output_root, "anatomy_tracking_format")
    tracks_dir = os.path.join(output_root, "anatomy_tracks")
    features_dir = os.path.join(output_root, "anatomy_features")
    anatomy_models_dir = os.path.join(output_root, "anatomy_classifier_results", "models")
    anatomy_reports_dir = os.path.join(output_root, "anatomy_classifier_results", "reports")
    ensemble_dir = os.path.join(output_root, "ensemble_results")
    comparison_dir = os.path.join(output_root, "model_comparison")

    detections_csv = os.path.join(tracking_format_dir, "detections.csv")
    tracks_csv = os.path.join(tracks_dir, "tracks.csv")
    features_csv = os.path.join(features_dir, "anatomy_features.csv")

    if not args.skip_audit:
        run_step(
            "Audit anatomy labels",
            [
                sys.executable,
                script_path("audit_tracking_labels.py"),
                "--index",
                args.index,
                "--output-dir",
                audit_dir,
            ],
            dry_run=args.dry_run,
        )

    run_step(
        "Convert LabelMe annotations",
        [
            sys.executable,
            script_path("convert_labelme_to_tracking_format.py"),
            "--index",
            args.index,
            "--output-dir",
            tracking_format_dir,
        ],
        dry_run=args.dry_run,
    )
    run_step(
        "Track anatomy",
        [
            sys.executable,
            script_path("track_anatomy.py"),
            "--detections-csv",
            detections_csv,
            "--output-dir",
            tracks_dir,
        ],
        dry_run=args.dry_run,
    )
    run_step(
        "Extract anatomy features and feedback",
        [
            sys.executable,
            script_path("extract_anatomy_features.py"),
            "--tracks-csv",
            tracks_csv,
            "--index",
            args.index,
            "--output-dir",
            features_dir,
        ],
        dry_run=args.dry_run,
    )
    run_step(
        "Train anatomy classifier",
        append_folds(
            [
                sys.executable,
                script_path("train_anatomy_classifier.py"),
                "--features-csv",
                features_csv,
                "--splits",
                args.splits,
                "--output-dir",
                anatomy_models_dir,
            ],
            args.folds,
        ),
        dry_run=args.dry_run,
    )
    run_step(
        "Evaluate anatomy classifier",
        append_folds(
            [
                sys.executable,
                script_path("eval_anatomy_classifier.py"),
                "--features-csv",
                features_csv,
                "--splits",
                args.splits,
                "--models-dir",
                anatomy_models_dir,
                "--reports-dir",
                anatomy_reports_dir,
            ],
            args.folds,
        ),
        dry_run=args.dry_run,
    )
    run_step(
        "Fuse hybrid and anatomy predictions",
        append_folds(
            [
                sys.executable,
                script_path("ensemble_predictions.py"),
                "--hybrid-reports-dir",
                args.hybrid_reports_dir,
                "--anatomy-reports-dir",
                anatomy_reports_dir,
                "--output-dir",
                ensemble_dir,
                "--hybrid-weight",
                str(args.hybrid_weight),
            ],
            args.folds,
        ),
        dry_run=args.dry_run,
    )
    run_step(
        "Compare final models",
        [
            sys.executable,
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
        dry_run=args.dry_run,
    )

    print("\nDone.", flush=True)
    print(f"Feedback file: {os.path.join(features_dir, 'anatomy_feedback.csv')}", flush=True)
    print(f"Model comparison: {os.path.join(comparison_dir, 'model_metrics_summary.csv\n')}", flush=True)


if __name__ == "__main__":
    main()
