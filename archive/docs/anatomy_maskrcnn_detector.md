# Anatomy Mask R-CNN Detector

This branch trains an automatic anatomy detector from the existing LabelMe
polygon annotations. It is separate from the deterministic anatomy tracker:
Mask R-CNN predicts anatomy masks and boxes from raw frames, then exports a
tracker-compatible `detections.csv`.

## Classes

- `vocal_cords`
- `arytenoids`
- `epiglottis`
- `esophagus`
- `endotracheal_tube`

Background is class id `0`.

## Train

Use the repo virtualenv because it has `torch` and `torchvision` installed:

```bash
.venv/bin/python archive/scripts/train_anatomy_maskrcnn.py \
  --fold 0 \
  --epochs 10 \
  --batch-size 2 \
  --output-dir archive/outputs/anatomy_detector
```

For a smoke test:

```bash
.venv/bin/python archive/scripts/train_anatomy_maskrcnn.py \
  --fold 0 \
  --epochs 1 \
  --batch-size 1 \
  --max-frames-per-video 2 \
  --val-max-frames-per-video 2 \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --no-pretrained \
  --output-dir archive/outputs/anatomy_detector_smoke
```

Main outputs:

- `archive/outputs/anatomy_detector/fold_0/maskrcnn_best.pt`
- `archive/outputs/anatomy_detector/fold_0/maskrcnn_last.pt`
- `archive/outputs/anatomy_detector/fold_0/training_log.csv`
- `archive/outputs/anatomy_detector/fold_0/training_config.json`

The default training split is video-level `fold_0/train` from
`data/videos/json_utils/splits_poly_50.json`. Validation uses `fold_0/val`.

## Predict Detections

Run detector inference and export predictions in the same schema consumed by
`archive/scripts/track_anatomy.py`:

```bash
.venv/bin/python archive/scripts/predict_anatomy_maskrcnn.py \
  --checkpoint archive/outputs/anatomy_detector/fold_0/maskrcnn_best.pt \
  --splits data/videos/json_utils/splits_poly_50.json \
  --fold 0 \
  --split val \
  --frame-source all \
  --score-threshold 0.5 \
  --output-dir archive/outputs/anatomy_detector/predictions/fold_0_val
```

Output:

- `archive/outputs/anatomy_detector/predictions/fold_0_val/detections.csv`

## Track Predicted Anatomy

Feed the predicted detections into the existing deterministic tracker:

```bash
.venv/bin/python archive/scripts/track_anatomy.py \
  --detections-csv archive/outputs/anatomy_detector/predictions/fold_0_val/detections.csv \
  --output-dir archive/outputs/anatomy_detector/predictions/fold_0_val_tracks
```

Then extract anatomy features from the predicted tracks:

```bash
.venv/bin/python archive/scripts/extract_anatomy_features.py \
  --tracks-csv archive/outputs/anatomy_detector/predictions/fold_0_val_tracks/tracks.csv \
  --index data/videos/json_utils/index_poly.json \
  --output-dir archive/outputs/anatomy_detector/predictions/fold_0_val_features
```

## Notes

- The detector should be validated by video split, not frame split, because
  adjacent frames are highly correlated.
- COCO pretrained Mask R-CNN weights are used by default. If weights are not
  cached and the environment cannot download them, use `--no-pretrained` for a
  smoke test or rerun in an environment with network access once to cache them.
- `epiglottis` has the fewest labeled instances, so expect class imbalance there.

## Full Detector + Hybrid Ensemble Run

Use this script when you want the automatic anatomy detector branch and the
hybrid SVM branch evaluated as one ensemble pipeline:

```bash
.venv/bin/python archive/scripts/run_detector_hybrid_ensemble.py \
  --folds 0 1 2 3 \
  --detector-epochs 10 \
  --detector-batch-size 2 \
  --predict-frame-source annotated \
  --hybrid-weight 0.5 \
  --output-root archive/outputs/detector_hybrid_ensemble
```

This runs:

- hybrid SVM training/evaluation through `src/run_SVM/run.py`
- fold-specific Mask R-CNN detector training
- detector prediction on each fold's train and validation videos
- existing deterministic anatomy tracking on predicted detections
- anatomy feature extraction from predicted tracks
- anatomy classifier training/evaluation using detector-derived features
- late fusion of hybrid and detector-anatomy probabilities
- final model comparison

If the hybrid SVM reports already exist and you only want to rerun the detector
ensemble branch:

```bash
.venv/bin/python archive/scripts/run_detector_hybrid_ensemble.py \
  --skip-hybrid \
  --hybrid-reports-dir runs/run_SVM/reports \
  --folds 0 1 2 3 \
  --detector-epochs 10 \
  --detector-batch-size 2 \
  --predict-frame-source annotated \
  --output-root archive/outputs/detector_hybrid_ensemble
```

For a quick wiring check without a real detector run:

```bash
.venv/bin/python archive/scripts/run_detector_hybrid_ensemble.py \
  --folds 0 \
  --skip-hybrid \
  --skip-ensemble \
  --detector-epochs 1 \
  --detector-batch-size 1 \
  --detector-max-frames-per-video 1 \
  --detector-val-max-frames-per-video 1 \
  --detector-max-train-samples 1 \
  --detector-max-val-samples 1 \
  --detector-no-pretrained \
  --detector-device cpu \
  --detector-min-size 128 \
  --detector-max-size 256 \
  --predict-frame-source annotated \
  --predict-frame-stride 999999 \
  --predict-score-threshold 0.0 \
  --predict-max-detections-per-frame 2 \
  --output-root archive/outputs/detector_hybrid_ensemble_smoke
```

Main outputs:

- `archive/outputs/detector_hybrid_ensemble/anatomy_classifier_results/reports/all_folds_results.csv`
- `archive/outputs/detector_hybrid_ensemble/ensemble_results/all_folds_results.csv`
- `archive/outputs/detector_hybrid_ensemble/model_comparison/model_metrics_summary.csv`
- `archive/outputs/detector_hybrid_ensemble/run_summary.json`
