from __future__ import annotations
import argparse
from pathlib import Path
try:
    from .utils import (
        DEFAULT_OUTPUT_CSV,
        REPO_ROOT,
        build_embedder,
        choose_device,
        collect_video_paths,
        load_models,
        predict_video,
        svm_config,
        validate_args,
        write_csv,
    )
except ImportError:
    from utils import (
        DEFAULT_OUTPUT_CSV,
        REPO_ROOT,
        build_embedder,
        choose_device,
        collect_video_paths,
        load_models,
        predict_video,
        svm_config,
        validate_args,
        write_csv,
    )

#Parsing arguments and running the main function.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the saved SVM + ResNet-18 hybrid model on .avi/.mp4 videos."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more video files or directories containing .avi/.mp4 files.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / svm_config.models_dir),
        help="Directory containing svm_fold_*.joblib files.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Optional explicit model path. Repeat to specify multiple SVM models.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=int(svm_config.frames_per_video_validation),
        help="Number of uniformly sampled frames per video.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frame batch size used while extracting ResNet-18 embeddings.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: cpu, mps, or cuda. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8, #if threshold < 0.8 the model will predict FAIL, otherwise PASS
        help="PASS threshold applied to the ensembled probability.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV path for saving predictions. Defaults to deploy/results/predictions.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = choose_device(args.device)
    resize_hw: tuple[int, int] = tuple(int(v) for v in svm_config.resize_hw)  # type: ignore

    # Dev print statements - uncomment if needed for debugging
    # print(f"\nUsing device: {device}")
    # print(f"Using resize: {resize_hw[0]}x{resize_hw[1]}")

    video_paths = collect_video_paths(args.inputs)
    models = load_models(args.model_dir, args.model_path)
    # print(f"Loaded {len(models)} SVM model(s)")

    embedder = build_embedder(device)
    results = []

    for idx, video_path in enumerate(video_paths, start=1):
        print(f"[{idx}/{len(video_paths)}] Predicting {video_path}")
        row = predict_video(
            video_path,
            embedder=embedder,
            models=models,
            device=device,
            resize_hw=resize_hw,
            frames_per_video=args.frames_per_video,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        results.append(row)

        confidence_obj = row.get("label_confidence", 0.0)
        confidence = (
            float(confidence_obj)
            if isinstance(confidence_obj, (int, float, str))
            else 0.0
        )
        print(
            f"Model is {confidence * 100:.2f}% confident that video [{row['video_name']}] "
            f"is a {row['predicted_label']}, please check .csv file for further statistics!"
        )

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else DEFAULT_OUTPUT_CSV.resolve()
    )
    write_csv(output_csv, results)
    print(f"Wrote predictions to {output_csv}\n")


if __name__ == "__main__":
    main()
