# Anatomy-Aware ETI Tracking Pipeline (╯°□°）╯
Run commands from the repo root. Use `python3` in this environment. 

## Final Ensemble Run
Use this command for the current final 50-video model:

```bash
python3 scripts/run.py
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

- `outputs/anatomy_features/anatomy_feedback.csv`
- `outputs/model_comparison/model_metrics_summary.csv`
- `outputs/model_comparison/failure_case_comparison.csv`

For faster reruns after the audit has already been checked:

```bash
python3 scripts/run.py --skip-audit
```

## Stage 1: Annotation Audit And Conversion
Audit LabelMe coverage and label consistency:

```bash
python3 scripts/audit_tracking_labels.py \
  --index data/videos/index_poly.json \
  --output-dir outputs/anatomy_label_audit
```

Convert LabelMe polygon or rectangle annotations into compact tracker inputs:

```bash
python3 scripts/convert_labelme_to_tracking_format.py \
  --index data/videos/index_poly.json \
  --output-dir outputs/anatomy_tracking_format
```

Outputs:

- `outputs/anatomy_label_audit/video_summary.csv`
- `outputs/anatomy_label_audit/audit_summary.json`
- `outputs/anatomy_tracking_format/detections.csv`

`detections.csv` is the only conversion file required for the baseline tracker.
COCO detector files are optional and can be generated later with `--write-coco`.
Detailed label-count CSVs are optional and can be generated with `--write-label-details`.

Track detections with simple per-class IoU and centroid association:

```bash
python3 scripts/track_anatomy.py \
  --detections-csv outputs/anatomy_tracking_format/detections.csv \
  --output-dir outputs/anatomy_tracks
```

Outputs:

- `outputs/anatomy_tracks/tracks.csv`

`tracks.csv` is intentionally compact. It keeps frame, class, bbox, centroid,
confidence, and track id fields, but drops raw polygon JSON and annotation paths.

## Stage 1/2: Anatomy Features And Classifier

Extract video-level features:

```bash
python3 scripts/extract_anatomy_features.py \
  --tracks-csv outputs/anatomy_tracks/tracks.csv \
  --index data/videos/index_poly.json \
  --output-dir outputs/anatomy_features
```

Train and evaluate the anatomy-only classifier on existing folds:

```bash
python3 scripts/train_anatomy_classifier.py \
  --features-csv outputs/anatomy_features/anatomy_features.csv \
  --splits data/videos/splits_poly_50.json \
  --output-dir outputs/anatomy_classifier_results/models

python3 scripts/eval_anatomy_classifier.py \
  --features-csv outputs/anatomy_features/anatomy_features.csv \
  --splits data/videos/splits_poly_50.json \
  --models-dir outputs/anatomy_classifier_results/models \
  --reports-dir outputs/anatomy_classifier_results/reports
```

Outputs:

- `outputs/anatomy_features/anatomy_features.csv`
- `outputs/anatomy_features/anatomy_feedback.csv`
- `outputs/anatomy_classifier_results/models/anatomy_classifier_fold_*.json`
- `outputs/anatomy_classifier_results/reports/fold_*_results.csv`
- `outputs/anatomy_classifier_results/reports/all_folds_results.csv`

## Stage 3: Fusion And Comparison

Late fusion with the current ResNet-18 + SVM hybrid reports:

```bash
python3 scripts/ensemble_predictions.py \
  --hybrid-reports-dir runs/run_SVM/res_eval/reports_50Poly_224x224 \
  --anatomy-reports-dir outputs/anatomy_classifier_results/reports \
  --output-dir outputs/ensemble_results \
  --hybrid-weight 0.5
```

Compare models:

```bash
python3 scripts/compare_models.py \
  --hybrid-results runs/run_SVM/res_eval/reports_50Poly_224x224/all_folds_results.csv \
  --anatomy-results outputs/anatomy_classifier_results/reports/all_folds_results.csv \
  --ensemble-results outputs/ensemble_results/all_folds_results.csv \
  --output-dir outputs/model_comparison
```

Outputs:

- `outputs/ensemble_results/all_folds_results.csv`
- `outputs/ensemble_results/all_folds_metrics.json`
- `outputs/model_comparison/model_metrics_summary.csv`
- `outputs/model_comparison/per_video_model_comparison.csv`
- `outputs/model_comparison/failure_case_comparison.csv`

## Notes

- The anatomy branch is parallel to the current hybrid pipeline.
- The detector/tracker path is annotation-derived and deterministic.
- The main feedback file is `outputs/anatomy_features/anatomy_feedback.csv`.
- The label normalizer currently maps variants such as `vocal_chords` to `vocal_cords`.
- The same `splits_poly_50.json` fold structure is used by default for anatomy classifier evaluation and fusion.
- The default hybrid comparison target is `runs/run_SVM/res_eval/reports_50Poly_224x224`.

## Models Used

- Hybrid branch: ResNet-18 feature extractor plus SVM classifier, using the saved 50-video SVM reports by default.
- Anatomy detector/tracker V1: annotation-derived detections plus deterministic IoU/centroid temporal association. This is not a learned ML tracker.
- Anatomy PASS/FAIL branch: NumPy logistic regression trained on video-level anatomy features.
- Ensemble branch: weighted late fusion by probability averaging.

(´｡• ᵕ •｡`) 
