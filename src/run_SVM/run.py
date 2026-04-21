import os
import sys
import time
#Ensure `<repo_root>/src` is on sys.path so we can import run_2DCNN, run_3DCNN, etc.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import torch
from run_SVM.extract_features import extract_fold
from run_SVM.train_svm import train_fold
from run_SVM.eval import evaluate_folds
from run_SVM.biomarker_eval import run_biomarker_eval
from run_SVM import config


def _format_seconds(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def time_training(fold: int) -> float:
    """Train one fold and return elapsed training time in seconds."""
    start = time.perf_counter()
    train_fold(fold)
    elapsed = time.perf_counter() - start
    print(f"Fold {fold}: SVM training completed in {_format_seconds(elapsed)}")
    return elapsed


#Main script to run the entire SVM pipeline
def main():
    pipeline_start = time.perf_counter()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    folds = list(range(int(config.kfolds)))
    train_times = []

    for fold in folds:
        print(f"Processing fold {fold}...")
        extract_fold(fold, device)
        train_times.append(time_training(fold))

    evaluate_folds(folds=folds)
    for fold in folds:
        run_biomarker_eval(fold=fold)
    run_biomarker_eval()

    total_train_time = sum(train_times)
    print(f"Total SVM training time: {_format_seconds(total_train_time)}")
    pipeline_elapsed = time.perf_counter() - pipeline_start
    print(f"Total SVM pipeline runtime: {_format_seconds(pipeline_elapsed)}")


if __name__ == "__main__":
    main()
