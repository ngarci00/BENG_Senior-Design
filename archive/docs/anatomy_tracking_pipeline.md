# Anatomy-Aware ETI Tracking Pipeline (╯°□°）╯
Run commands from the repo root. Use `python3` in this environment. 

## Final Ensemble Run
Use this command for the current final 50-video model:

```bash
python3 archive/scripts/run.py
```

This runs:

- annotation audit
- LabelMe conversion
- baseline anatomy tracking
- anatomy feature and feedback extraction
- anatomy classifier training/evaluation
- late fusion with the saved ResNet-18 + SVM hybrid reports
- final model comparison

Main outputs:

- `archive/outputs/anatomy_features/anatomy_feedback.csv`
- `archive/outputs/model_comparison/model_metrics_summary.csv`
- `archive/outputs/model_comparison/failure_case_comparison.csv`

For faster reruns after the audit has already been checked:

```bash
python3 archive/scripts/run.py --skip-audit
```

## Stage 1: Annotation Audit And Conversion
Audit LabelMe coverage and label consistency:

```bash
python3 archive/scripts/audit_tracking_labels.py \
  --index data/videos/json_utils/index_poly.json \
  --output-dir archive/outputs/anatomy_label_audit
```

Convert LabelMe polygon or rectangle annotations into compact tracker inputs:

```bash
python3 archive/scripts/convert_labelme_to_tracking_format.py \
  --index data/videos/json_utils/index_poly.json \
  --output-dir archive/outputs/anatomy_tracking_format
```

Outputs:

- `archive/outputs/anatomy_label_audit/video_summary.csv`
- `archive/outputs/anatomy_label_audit/audit_summary.json`
- `archive/outputs/anatomy_tracking_format/detections.csv`

`detections.csv` is the only conversion file required for the baseline tracker.
COCO detector files are optional and can be generated later with `--write-coco`.
Detailed label-count CSVs are optional and can be generated with `--write-label-details`.

Track detections with simple per-class IoU and centroid association:

```bash
python3 archive/scripts/track_anatomy.py \
  --detections-csv archive/outputs/anatomy_tracking_format/detections.csv \
  --output-dir archive/outputs/anatomy_tracks
```

Outputs:

- `archive/outputs/anatomy_tracks/tracks.csv`

`tracks.csv` is intentionally compact. It keeps frame, class, bbox, centroid,
confidence, and track id fields, but drops raw polygon JSON and annotation paths.

## Stage 1/2: Anatomy Features And Classifier

Extract video-level features:

```bash
python3 archive/scripts/extract_anatomy_features.py \
  --tracks-csv archive/outputs/anatomy_tracks/tracks.csv \
  --index data/videos/json_utils/index_poly.json \
  --output-dir archive/outputs/anatomy_features
```

Train and evaluate the anatomy-only classifier on existing folds:

```bash
python3 archive/scripts/train_anatomy_classifier.py \
  --features-csv archive/outputs/anatomy_features/anatomy_features.csv \
  --splits data/videos/json_utils/splits_poly_50.json \
  --output-dir archive/outputs/anatomy_classifier_results/models

python3 archive/scripts/eval_anatomy_classifier.py \
  --features-csv archive/outputs/anatomy_features/anatomy_features.csv \
  --splits data/videos/json_utils/splits_poly_50.json \
  --models-dir archive/outputs/anatomy_classifier_results/models \
  --reports-dir archive/outputs/anatomy_classifier_results/reports
```

Outputs:

- `archive/outputs/anatomy_features/anatomy_features.csv`
- `archive/outputs/anatomy_features/anatomy_feedback.csv`
- `archive/outputs/anatomy_classifier_results/models/anatomy_classifier_fold_*.json`
- `archive/outputs/anatomy_classifier_results/reports/fold_*_results.csv`
- `archive/outputs/anatomy_classifier_results/reports/all_folds_results.csv`

## Stage 3: Fusion And Comparison

Late fusion with the current ResNet-18 + SVM hybrid reports:

```bash
python3 archive/scripts/ensemble_predictions.py \
  --hybrid-reports-dir runs/run_HYBRID/reports \
  --anatomy-reports-dir archive/outputs/anatomy_classifier_results/reports \
  --output-dir archive/outputs/ensemble_results \
  --hybrid-weight 0.5
```

Compare models:

```bash
python3 archive/scripts/compare_models.py \
  --hybrid-results runs/run_HYBRID/reports/all_folds_results.csv \
  --anatomy-results archive/outputs/anatomy_classifier_results/reports/all_folds_results.csv \
  --ensemble-results archive/outputs/ensemble_results/all_folds_results.csv \
  --output-dir archive/outputs/model_comparison
```

Outputs:

- `archive/outputs/ensemble_results/all_folds_results.csv`
- `archive/outputs/ensemble_results/all_folds_metrics.json`
- `archive/outputs/model_comparison/model_metrics_summary.csv`
- `archive/outputs/model_comparison/per_video_model_comparison.csv`
- `archive/outputs/model_comparison/failure_case_comparison.csv`

## Notes

- The anatomy branch is parallel to the current hybrid pipeline.
- The detector/tracker path is annotation-derived and deterministic.
- The main feedback file is `archive/outputs/anatomy_features/anatomy_feedback.csv`.
- The label normalizer currently maps variants such as `vocal_chords` to `vocal_cords`.
- The same `splits_poly_50.json` fold structure is used by default for anatomy classifier evaluation and fusion.
- The default hybrid comparison target is `runs/run_HYBRID/reports`.

## Models Used

- Hybrid branch: ResNet-18 feature extractor plus SVM classifier, using the saved 50-video SVM reports by default.
- Anatomy detector/tracker V1: annotation-derived detections plus deterministic IoU/centroid temporal association. This is not a learned ML tracker.
- Anatomy PASS/FAIL branch: NumPy logistic regression trained on video-level anatomy features.
- Ensemble branch: weighted late fusion by probability averaging.

(´｡• ᵕ •｡`) 
