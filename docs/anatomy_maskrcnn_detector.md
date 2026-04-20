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
.venv/bin/python scripts/train_anatomy_maskrcnn.py \
  --fold 0 \
  --epochs 10 \
  --batch-size 2 \
  --output-dir outputs/anatomy_detector
```

For a smoke test:

```bash
.venv/bin/python scripts/train_anatomy_maskrcnn.py \
  --fold 0 \
  --epochs 1 \
  --batch-size 1 \
  --max-frames-per-video 2 \
  --val-max-frames-per-video 2 \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --no-pretrained \
  --output-dir outputs/anatomy_detector_smoke
```

Main outputs:

- `outputs/anatomy_detector/fold_0/maskrcnn_best.pt`
- `outputs/anatomy_detector/fold_0/maskrcnn_last.pt`
- `outputs/anatomy_detector/fold_0/training_log.csv`
- `outputs/anatomy_detector/fold_0/training_config.json`

The default training split is video-level `fold_0/train` from
`data/videos/splits_poly_50.json`. Validation uses `fold_0/val`.

## Predict Detections

Run detector inference and export predictions in the same schema consumed by
`scripts/track_anatomy.py`:

```bash
.venv/bin/python scripts/predict_anatomy_maskrcnn.py \
  --checkpoint outputs/anatomy_detector/fold_0/maskrcnn_best.pt \
  --splits data/videos/splits_poly_50.json \
  --fold 0 \
  --split val \
  --frame-source all \
  --score-threshold 0.5 \
  --output-dir outputs/anatomy_detector/predictions/fold_0_val
```

Output:

- `outputs/anatomy_detector/predictions/fold_0_val/detections.csv`

## Track Predicted Anatomy

Feed the predicted detections into the existing deterministic tracker:

```bash
.venv/bin/python scripts/track_anatomy.py \
  --detections-csv outputs/anatomy_detector/predictions/fold_0_val/detections.csv \
  --output-dir outputs/anatomy_detector/predictions/fold_0_val_tracks
```

Then extract anatomy features from the predicted tracks:

```bash
.venv/bin/python scripts/extract_anatomy_features.py \
  --tracks-csv outputs/anatomy_detector/predictions/fold_0_val_tracks/tracks.csv \
  --index data/videos/index_poly.json \
  --output-dir outputs/anatomy_detector/predictions/fold_0_val_features
```

## Notes

- The detector should be validated by video split, not frame split, because
  adjacent frames are highly correlated.
- COCO pretrained Mask R-CNN weights are used by default. If weights are not
  cached and the environment cannot download them, use `--no-pretrained` for a
  smoke test or rerun in an environment with network access once to cache them.
- `epiglottis` has the fewest labeled instances, so expect it have some class imbalances.
