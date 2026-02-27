import os
import sys

# Ensure `<repo_root>/src` is on sys.path so we can import run_2DCNN, run_3DCNN, etc.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
from extract_features import extract_fold
from train_svm import train_fold
from eval import eval_fold
import config
#Main script to run the entire SVM pipeline
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    for fold in range(config.kfolds):
        print(f"Processing fold {fold}...")
        extract_fold(fold, device)
        train_fold(fold)
        r = eval_fold(fold)
if __name__ == "__main__":
    main()