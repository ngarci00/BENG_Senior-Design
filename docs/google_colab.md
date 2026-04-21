# Google Colab Guide

This repository can run in Google Colab, but the data layout matters more than the code.

Current repo sizes on disk:

- `data/videos`: about 6.8 GB
- `runs`: about 821 MB
- `outputs`: about 112 MB

The recommended Colab workflow is:

1. Use a GPU runtime.
2. Clone only the code into `/content`.
3. Keep `data/` and large run artifacts in Google Drive.
4. Point the cloned repo at those Drive folders with symlinks.

## Runtime

In Colab, select `Runtime -> Change runtime type -> T4 GPU` or `L4 GPU`.

Then verify the runtime:

```python
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

## Recommended Setup

Run this in a Colab cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repository without downloading large LFS objects up front:

```bash
%cd /content
!apt-get -qq update
!apt-get -qq install git-lfs
!git lfs install
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ngarci00/BENG_Senior-Design.git
%cd /content/BENG_Senior-Design
!pip install -q -r requirements-colab.txt
```

## Drive Layout

Put your large folders somewhere in Drive, for example:

```text
MyDrive/BENG_Senior-Design/data
MyDrive/BENG_Senior-Design/runs
MyDrive/BENG_Senior-Design/outputs
```

Then attach them to the cloned repo:

```bash
%cd /content/BENG_Senior-Design
!rm -rf data runs outputs
!ln -s /content/drive/MyDrive/BENG_Senior-Design/data data
!ln -s /content/drive/MyDrive/BENG_Senior-Design/runs runs
!ln -s /content/drive/MyDrive/BENG_Senior-Design/outputs outputs
!ls data/videos | head
```

If you do not already have `runs/` or `outputs/` in Drive, create them first:

```bash
!mkdir -p /content/drive/MyDrive/BENG_Senior-Design/runs
!mkdir -p /content/drive/MyDrive/BENG_Senior-Design/outputs
```

## Optional Full Clone

If you want Colab to download the LFS-tracked assets directly from GitHub instead of using Drive:

```bash
%cd /content
!apt-get -qq update
!apt-get -qq install git-lfs
!git lfs install
!git clone https://github.com/ngarci00/BENG_Senior-Design.git
%cd /content/BENG_Senior-Design
!git lfs pull
!pip install -q -r requirements-colab.txt
```

Use this only if you actually want the full download. It is much slower and more fragile than Drive-backed data.

## Run Commands

From the repo root in Colab:

### 3DCNN

```bash
%cd /content/BENG_Senior-Design
!python src/run_3DCNN/run.py
```

### Hybrid SVM

```bash
%cd /content/BENG_Senior-Design
!python src/run_SVM/run.py
```

### Anatomy Tracking + Classifier + Ensemble

```bash
%cd /content/BENG_Senior-Design
!python scripts/run.py --skip-audit
```

This wrapper expects an existing hybrid report directory at
`runs/run_SVM/res_eval/reports_50Poly_224x224` unless you pass
`--hybrid-reports-dir` yourself.

### Detector + Hybrid Ensemble

This is the heaviest pipeline. Start with a reduced run first:

```bash
%cd /content/BENG_Senior-Design
!python scripts/run_detector_hybrid_ensemble.py \
  --folds 0 \
  --detector-epochs 1 \
  --detector-batch-size 2 \
  --detector-max-train-samples 32 \
  --detector-max-val-samples 16 \
  --detector-val-max-frames-per-video 16 \
  --predict-frame-source annotated
```

## Save New Outputs Back To Drive

If `outputs/` and `runs/` are symlinked to Drive as shown above, results are already saved persistently.

If you run without those symlinks, copy results back manually before the runtime disconnects.

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'`
  Colab runtime is missing PyTorch. Install a compatible `torch` and `torchvision` pair before running the repo.

- `FileNotFoundError` under `data/videos/...`
  Your Drive folder is not mounted or the symlink target is wrong.

- `cuda: False`
  The notebook is on a CPU runtime. Switch the runtime type to GPU and reconnect.

- Runtime disconnects during long training
  Run a single fold first, reduce detector sample counts, and keep `runs/` and `outputs/` on Drive.
