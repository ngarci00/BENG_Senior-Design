import os
import sys

# Ensure `<repo_root>/src` is on sys.path so `biomarker_driven_model` can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from biomarker_driven_model.extract_features import extract_fold
from biomarker_driven_model.train import train_fold
from biomarker_driven_model.eval import eval_fold, write_aggregate_reports
from biomarker_driven_model import config


def main():
    all_rows = []
    all_metrics = []
    for fold in range(config.kfolds):
        print("Processing fold", fold)
        extract_fold(fold)
        train_fold(fold)
        rows, metrics = eval_fold(fold)
        all_rows.extend(rows)
        all_metrics.append(metrics)
        print(
            "Fold",
            fold,
            "Accuracy:",
            f"{metrics['accuracy']:.3f}",
            "F1:",
            f"{metrics['f1_score']:.3f}",
            "n_videos:",
            metrics["n_videos"],
        )

    write_aggregate_reports(all_rows, all_metrics)


if __name__ == "__main__":
    main()
