#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from anatomy_detection import config
from anatomy_detection.dataset import LabelMeMaskDataset, collate_detection_batch, target_classes_from_args
from anatomy_detection.model import build_maskrcnn
from anatomy_tracking.io import ensure_dir, load_index, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN anatomy detector from LabelMe polygons.")
    parser.add_argument("--index", default=config.DEFAULT_INDEX_JSON)
    parser.add_argument("--splits", default=config.DEFAULT_SPLITS_JSON)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-classes", nargs="*", default=config.TARGET_CLASSES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from COCO Mask R-CNN weights")
    parser.add_argument("--allow-random-fallback", action="store_true", help="Fall back to random weights if pretrained loading fails")
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--val-max-frames-per-video", type=int, default=80)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_targets(targets: Iterable[Dict], device: torch.device) -> List[Dict]:
    moved = []
    for target in targets:
        moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return moved


def mean_loss(losses: List[float]) -> float:
    finite = [value for value in losses if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def train_one_epoch(model, loader, optimizer, device: torch.device, epoch: int) -> Dict[str, float]:
    model.train()
    batch_losses: List[float] = []
    component_sums: Dict[str, float] = {}
    start = time.perf_counter()

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_float = float(loss.detach().cpu())
        batch_losses.append(loss_float)
        for key, value in loss_dict.items():
            component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().cpu())

        if batch_idx % 20 == 0 or batch_idx == len(loader):
            print(f"epoch {epoch} batch {batch_idx}/{len(loader)} loss={mean_loss(batch_losses):.4f}", flush=True)

    metrics = {"train_loss": mean_loss(batch_losses), "epoch_seconds": time.perf_counter() - start}
    for key, total in sorted(component_sums.items()):
        metrics[f"train_{key}"] = total / max(len(loader), 1)
    return metrics


@torch.no_grad()
def validation_loss(model, loader, device: torch.device) -> Dict[str, float]:
    was_training = model.training
    model.train()
    batch_losses: List[float] = []
    component_sums: Dict[str, float] = {}

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())
        batch_losses.append(float(loss.detach().cpu()))
        for key, value in loss_dict.items():
            component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().cpu())

    model.train(was_training)
    metrics = {"val_loss": mean_loss(batch_losses)}
    for key, total in sorted(component_sums.items()):
        metrics[f"val_{key}"] = total / max(len(loader), 1)
    return metrics


def save_checkpoint(path: str, model, optimizer, epoch: int, args: argparse.Namespace, target_classes: List[str], metrics: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "target_classes": target_classes,
            "class_to_id": {class_name: idx + 1 for idx, class_name in enumerate(target_classes)},
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def append_log(path: str, row: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint(path: str, model, optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint.get("epoch", -1)) + 1


def build_model(args: argparse.Namespace, num_classes: int):
    try:
        return build_maskrcnn(
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            min_size=args.min_size,
            max_size=args.max_size,
            trainable_backbone_layers=args.trainable_backbone_layers,
        )
    except Exception:
        if not args.allow_random_fallback or args.no_pretrained:
            raise
        print("Pretrained Mask R-CNN weights could not be loaded; falling back to random initialization.", flush=True)
        return build_maskrcnn(
            num_classes=num_classes,
            pretrained=False,
            min_size=args.min_size,
            max_size=args.max_size,
            trainable_backbone_layers=args.trainable_backbone_layers,
        )


def main() -> None:
    args = parse_args()
    target_classes = target_classes_from_args(args.target_classes)
    run_dir = os.path.join(args.output_dir, f"fold_{args.fold}")
    ensure_dir(run_dir)

    device = pick_device(args.device)
    print(f"Using device: {device}", flush=True)

    train_ds = LabelMeMaskDataset(
        index_path=args.index,
        splits_path=args.splits,
        fold=args.fold,
        split="train",
        target_classes=target_classes,
        max_frames_per_video=args.max_frames_per_video,
        frame_stride=args.frame_stride,
        hflip_prob=args.hflip_prob,
    )
    val_ds = LabelMeMaskDataset(
        index_path=args.index,
        splits_path=args.splits,
        fold=args.fold,
        split="val",
        target_classes=target_classes,
        max_frames_per_video=args.val_max_frames_per_video,
        frame_stride=args.frame_stride,
        hflip_prob=0.0,
    )
    if args.max_train_samples is not None:
        train_ds.samples = train_ds.samples[: int(args.max_train_samples)]
    if args.max_val_samples is not None:
        val_ds.samples = val_ds.samples[: int(args.max_val_samples)]
    if not train_ds:
        first_meta = next(iter(load_index(args.index)), {})
        raise RuntimeError(
            "Training dataset is empty. This usually means the index file points to annotation/frame directories "
            "that do not exist on the current machine. Example resolved paths: "
            f"frames_dir={first_meta.get('frames_dir', '')!r}, ann_dir={first_meta.get('ann_dir', '')!r}"
        )
    if not val_ds:
        first_meta = next(iter(load_index(args.index)), {})
        raise RuntimeError(
            "Validation dataset is empty. This usually means the index file points to annotation/frame directories "
            "that do not exist on the current machine. Example resolved paths: "
            f"frames_dir={first_meta.get('frames_dir', '')!r}, ann_dir={first_meta.get('ann_dir', '')!r}"
        )

    print(f"Train frames: {len(train_ds)} | Val frames: {len(val_ds)}", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )

    model = build_model(args, num_classes=len(target_classes) + 1).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    write_json(
        os.path.join(run_dir, "training_config.json"),
        {
            "args": vars(args),
            "target_classes": target_classes,
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "device": str(device),
        },
    )

    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validation_loss(model, val_loader, device)
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        append_log(os.path.join(run_dir, "training_log.csv"), row)
        print(
            f"epoch {epoch} done train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f}",
            flush=True,
        )

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(os.path.join(run_dir, "maskrcnn_last.pt"), model, optimizer, epoch, args, target_classes, row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_checkpoint(os.path.join(run_dir, "maskrcnn_best.pt"), model, optimizer, epoch, args, target_classes, row)

    print(f"Training complete. Best checkpoint: {os.path.join(run_dir, 'maskrcnn_best.pt')}", flush=True)


if __name__ == "__main__":
    main()
