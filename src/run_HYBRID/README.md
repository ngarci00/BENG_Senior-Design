# `2D CNN + SVM Model`  ⸜(｡˃ ᵕ ˂ )⸝♡ 

This is the active video-classification pipeline in the repository!

It uses:

- Aampled video frames from the dataset index/split JSON files
- A pretrained ResNet-18 feature extractor to embed frames
- Mean pooling to turn frame embeddings into one video embedding
- An SVM classifier for PASS vs FAIL prediction

## Main Files

- `config.py`: dataset paths, split file, resize, frame sampling, and run directories
- `dataset.py`: loads one video at a time as a `(T, C, H, W)` clip tensor
- `extract_features.py`: converts each video into one embedding and saves cached fold features
- `train_svm.py`: trains one SVM per fold and saves `svm_fold_*.joblib`
- `eval.py`: evaluates saved fold models and writes reports/plots
- `biomarker_eval.py`: post-processing analysis by biomarker presence
- `run.py`: full end-to-end training/evaluation entrypoint

## Pipeline Order

1. Set the dataset/split settings in `config.py`.
2. Run `run.py` to extract features, train the fold models, and evaluate them.
3. Review outputs in `runs/run_HYBRID/`.
4. If needed, run `biomarker_eval.py` for the biomarker-specific breakdown.

## Outputs

The main outputs are written under:

- `runs/run_HYBRID/svm_features/`
- `runs/run_HYBRID/models/`
- `runs/run_HYBRID/reports/`

## Notes

- This is the active pipeline. Older standalone 2D CNN and 3D CNN code now lives in `src/archived_models/`.
- Make sure the JSON index and split paths in `config.py` match the dataset you want to use before starting a run.
