# Train a 3D CNN (torchvision r3d_18) using k-fold CV splits from data/videos/splits.json

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VideoClipDataset, load_image


# -------------------------
# Model
# -------------------------

def build_3dcnn(num_classes: int = 1) -> nn.Module:
    """Build a 3D CNN (r3d_18) with a binary head.

    Returns logits of shape (B, 1). Use BCEWithLogitsLoss.
    """
    try:
        import torchvision
        from torchvision.models.video import r3d_18

        # torchvision API differs across versions
        try:
            weights = torchvision.models.video.R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
        except Exception:
            try:
                model = r3d_18(pretrained=True)
            except Exception:
                model = r3d_18(pretrained=False)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    except Exception as e:
        raise RuntimeError(
            "Failed to build torchvision r3d_18. Ensure torchvision is installed and supports video models."
        ) from e


# -------------------------
# Metrics (no sklearn needed)
# -------------------------

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _confusion_counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    c = _confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # sensitivity for PASS class
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        **c,
    }


# -------------------------
# Video-level evaluation (multi-clip aggregation)
# -------------------------

@torch.no_grad()
def predict_video_probability(
    model: nn.Module,
    video_item: Dict[str, Any],
    device: str,
    clip_len: int,
    resize_hw: Tuple[int, int],
    eval_clips_per_video: int = 5,
) -> float:
    """Return mean PASS probability over multiple clips for a single video."""
    model.eval()

    frames_dir = Path(video_item["frames_dir"])
    frames = video_item["frames"]
    T = len(frames)

    if T == 0:
        raise RuntimeError(f"{video_item['video_id']}: no frames in index.json")

    if T <= clip_len:
        starts = [0]
    else:
        max_start = T - clip_len
        if eval_clips_per_video <= 1:
            starts = [max_start // 2]
        else:
            starts = [round(i * max_start / (eval_clips_per_video - 1)) for i in range(eval_clips_per_video)]

    probs: List[float] = []
    for s in starts:
        chosen = frames[s : s + clip_len]
        if len(chosen) < clip_len:
            chosen = chosen + [chosen[-1]] * (clip_len - len(chosen))

        clip = [load_image(frames_dir / fn, resize_hw) for fn in chosen]
        x = torch.stack(clip, dim=0).permute(1, 0, 2, 3).contiguous()  # (C,T,H,W)
        x = x.unsqueeze(0).to(device)  # (1,C,T,H,W)

        logits = model(x)  # (1,1)
        p = torch.sigmoid(logits).item()
        probs.append(float(p))

    return float(sum(probs) / len(probs))


@torch.no_grad()
def evaluate_fold(
    model: nn.Module,
    index_items: List[Dict[str, Any]],
    val_video_ids: List[str],
    device: str,
    clip_len: int,
    resize_hw: Tuple[int, int],
    eval_clips_per_video: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    id_to_item = {x["video_id"]: x for x in index_items}

    y_true: List[int] = []
    y_pred: List[int] = []

    for vid in val_video_ids:
        item = id_to_item[vid]
        p = predict_video_probability(
            model=model,
            video_item=item,
            device=device,
            clip_len=clip_len,
            resize_hw=resize_hw,
            eval_clips_per_video=eval_clips_per_video,
        )
        pred = 1 if p >= threshold else 0
        y_true.append(int(item["label"]))
        y_pred.append(pred)

    return classification_metrics(y_true, y_pred)


# -------------------------
# Training
# -------------------------

def train_one_fold(
    fold: int,
    val_video_ids: List[str],
    index_items: List[Dict[str, Any]],
    out_dir: Path,
    device: str,
    clip_len: int = 16,
    resize_hw: Tuple[int, int] = (112, 112),
    clips_per_video: int = 20,
    batch_size: int = 2,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    eval_clips_per_video: int = 5,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = VideoClipDataset(
        fold=fold,
        split="train",
        clip_len=clip_len,
        resize_hw=resize_hw,
        clips_per_video=clips_per_video,
        only_annotated_frames=False,
        seed=42,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_3dcnn(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_ckpt = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y, _vid in dl_train:
            x = x.to(device)  # (B,C,T,H,W)
            y = y.float().to(device).view(-1, 1)  # (B,1)

            logits = model(x)  # (B,1)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())

        train_loss = running / max(len(dl_train), 1)

        metrics = evaluate_fold(
            model=model,
            index_items=index_items,
            val_video_ids=val_video_ids,
            device=device,
            clip_len=clip_len,
            resize_hw=resize_hw,
            eval_clips_per_video=eval_clips_per_video,
            threshold=0.5,
        )

        print(
            f"[fold {fold}] epoch {epoch:02d}/{epochs} "
            f"loss={train_loss:.4f} val_f1={metrics['f1']:.3f} "
            f"acc={metrics['accuracy']:.3f} (tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']})"
        )

        # Persist best model by F1
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)

        # Persist last epoch metrics (overwrites each epoch)
        (out_dir / "metrics_last_epoch.json").write_text(
            json.dumps({"epoch": epoch, "train_loss": train_loss, **metrics}, indent=2)
        )

    # Reload best and compute final metrics
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    final_metrics = evaluate_fold(
        model=model,
        index_items=index_items,
        val_video_ids=val_video_ids,
        device=device,
        clip_len=clip_len,
        resize_hw=resize_hw,
        eval_clips_per_video=eval_clips_per_video,
        threshold=0.5,
    )

    (out_dir / "metrics_best.json").write_text(json.dumps(final_metrics, indent=2))
    return final_metrics


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index_path = Path("data/videos/index.json")
    splits_path = Path("data/videos/splits.json")

    run_root = Path("runs/kfold_r3d18")
    run_root.mkdir(parents=True, exist_ok=True)

    index_items: List[Dict[str, Any]] = json.loads(index_path.read_text())
    splits: Dict[str, Dict[str, List[str]]] = json.loads(splits_path.read_text())

    all_fold_metrics: Dict[str, Dict[str, float]] = {}

    for fold_key, split in splits.items():
        fold = int(fold_key.split("_")[-1])
        out_dir = run_root / fold_key

        m = train_one_fold(
            fold=fold,
            val_video_ids=split["val"],
            index_items=index_items,
            out_dir=out_dir,
            device=device,
            clip_len=16,
            resize_hw=(112, 112),
            clips_per_video=20,
            batch_size=2,
            epochs=10,
            lr=3e-4,
            weight_decay=1e-2,
            eval_clips_per_video=5,
        )
        all_fold_metrics[fold_key] = m

    # Summary mean/std
    keys = ["accuracy", "precision", "recall", "specificity", "f1"]
    summary: Dict[str, Dict[str, float]] = {}

    for k in keys:
        vals = [all_fold_metrics[f][k] for f in all_fold_metrics]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        std = math.sqrt(var)
        summary[k] = {"mean": float(mean), "std": float(std)}

    (run_root / "summary.json").write_text(
        json.dumps({"folds": all_fold_metrics, "summary": summary}, indent=2)
    )

    print("K-fold summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()