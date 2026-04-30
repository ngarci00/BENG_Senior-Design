from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import imageio
import joblib
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_SVM import config as svm_config
from run_SVM.feature_extractor import ResNet18Embedder


SUPPORTED_VIDEO_EXTS = {".avi", ".mp4"}
INT_TO_LABEL = {0: "FAIL", 1: "PASS"}
DEFAULT_RESULTS_DIR = REPO_ROOT / "deploy" / "results"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / "predictions.csv"
LoadedModel = Any
PredictionRow = Dict[str, object]


def choose_device(name: str | None) -> str:
    if name:
        device = str(name).strip().lower()
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device: {name}")
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def validate_args(args: argparse.Namespace) -> None:
    if args.frames_per_video < 1:
        raise ValueError("--frames-per-video must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0.")


def collect_video_paths(inputs: Sequence[str]) -> List[Path]:
    video_paths: List[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
            )

        for candidate in candidates:
            if candidate.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
                continue
            if candidate not in seen:
                seen.add(candidate)
                video_paths.append(candidate)

    if not video_paths:
        raise RuntimeError("No .avi or .mp4 files were found in the provided inputs.")

    video_paths.sort()
    return video_paths


def _model_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"svm_fold_(\d+)\.joblib$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def load_models(model_dir: str, explicit_model_paths: Sequence[str]) -> List[tuple[Path, LoadedModel]]:
    if explicit_model_paths:
        model_paths = [Path(p).expanduser().resolve() for p in explicit_model_paths]
    else:
        root = Path(model_dir).expanduser().resolve()
        model_paths = sorted(root.glob("svm_fold_*.joblib"), key=_model_sort_key)

    if not model_paths:
        raise RuntimeError("No SVM model files were found.")

    missing = [path for path in model_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model files:\n{missing_text}")

    return [(path, joblib.load(path)) for path in model_paths]


def build_embedder(device: str) -> ResNet18Embedder:
    try:
        embedder = ResNet18Embedder(pretrained=bool(svm_config.use_pretrained_backbone)).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the ResNet-18 feature extractor. "
            "Make sure the torchvision ResNet-18 ImageNet weights are available locally."
        ) from exc
    embedder.eval()
    return embedder


def uniform_sample_indices(n_frames: int, k: int) -> List[int]:
    if n_frames <= 0:
        raise RuntimeError("Video has no frames.")
    if k <= 1:
        return [0]
    return [int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)]


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Unexpected frame shape: {arr.shape}")

    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0


def sample_video_frames(
    video_path: Path,
    frames_per_video: int,
    resize_hw: tuple[int, int],
) -> tuple[torch.Tensor, int, List[int]]:
    reader = imageio.get_reader(str(video_path))
    try:
        n_frames = len(reader)
        sampled_indices = uniform_sample_indices(n_frames, frames_per_video)
        frames = [frame_to_tensor(reader.get_data(index)) for index in sampled_indices]
    finally:
        reader.close()

    x = torch.stack(frames, dim=0)
    x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
    return x.contiguous(), n_frames, sampled_indices


@torch.no_grad()
def build_video_embedding(
    embedder: ResNet18Embedder,
    frames: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    batches: List[torch.Tensor] = []

    for start in range(0, frames.shape[0], batch_size):
        batch = frames[start : start + batch_size].to(device)
        batches.append(embedder(batch).cpu())

    frame_embeddings = torch.cat(batches, dim=0)
    return frame_embeddings.mean(dim=0, keepdim=True).numpy()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def model_probability(model: LoadedModel, embedding: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        try:
            return float(model.predict_proba(embedding)[0, 1])
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(embedding), dtype=float).reshape(-1)
        return float(sigmoid(scores)[0])
    raise RuntimeError("Model does not support predict_proba or decision_function.")


def predict_video(
    video_path: Path,
    *,
    embedder: ResNet18Embedder,
    models: Sequence[tuple[Path, LoadedModel]],
    device: str,
    resize_hw: tuple[int, int],
    frames_per_video: int,
    batch_size: int,
    threshold: float,
) -> PredictionRow:
    frames, n_total_frames, sampled_indices = sample_video_frames(
        video_path=video_path,
        frames_per_video=frames_per_video,
        resize_hw=resize_hw,
    )
    embedding = build_video_embedding(
        embedder=embedder,
        frames=frames,
        device=device,
        batch_size=batch_size,
    )

    per_model_pass_probs = {
        model_path.name: model_probability(model, embedding) for model_path, model in models
    }

    pass_prob = float(np.mean(list(per_model_pass_probs.values())))
    fail_prob = 1.0 - pass_prob
    predicted_label_int = int(pass_prob >= threshold)
    predicted_label_name = INT_TO_LABEL[predicted_label_int]
    label_confidence = pass_prob if predicted_label_int == 1 else fail_prob

    return {
        # "video_path": str(video_path),
        "video_name": video_path.name,
        "predicted_label": predicted_label_name,
        # "predicted_label_int": predicted_label_int,
        "pass_prob": np.round(pass_prob*100,2), #Convert to percentage
        "fail_prob": np.round(fail_prob*100,2), #Convert to percentage
        "label_confidence": label_confidence,
        "threshold": threshold, #The threshold used to determine PASS vs FAIL
        # "n_models": len(models), #Number of SVM folds used in the ensemble
        "n_total_frames": n_total_frames,
        # "n_sampled_frames": len(sampled_indices),
        # "sampled_frame_indices": ";".join(str(i) for i in sampled_indices),
        # "device": device,
        # "resize_hw": f"{resize_hw[0]}x{resize_hw[1]}",
        # "per_model_pass_probs": json.dumps(per_model_pass_probs, sort_keys=True),
    }


def write_csv(path: Path, rows: List[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
