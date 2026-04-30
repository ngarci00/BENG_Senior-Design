# Deploy

How to locally run our 2D CNN + SVM model

## Run One Video
```bash
.venv/bin/python deploy/ETI_classifier.py path/to/new_video.avi
```

`.mp4` works the same way:

```bash
.venv/bin/python deploy/ETI_classifier.py path/to/new_video.mp4
```

## Run A Folder
```bash
.venv/bin/python deploy/ETI_classifier.py path/to/video_folder --output-csv deploy/predictions.csv
```

The script will:

- Load `runs/run_SVM/models/svm_fold_*.joblib`
- Sample frames uniformly from each video
- Resize frames to the current SVM config resolution
- Build ResNet-18 video embeddings by mean-pooling frame embeddings
- Average the fold-model probabilities into one final PASS/FAIL score
- Always write a CSV file of the results

## Please Note 

- Use the repo virtualenv: `.venv/bin/python`.
- The current script supports `.avi` and `.mp4`.
- If `--output-csv` is omitted, results are written to `deploy/results/predictions.csv`.
- The `deploy/results/` directory is created automatically.
- It uses the existing ImageNet-pretrained ResNet-18 feature extractor path from `src/run_SVM`.
- The torchvision ResNet-18 weights need to be available locally on the machine.
