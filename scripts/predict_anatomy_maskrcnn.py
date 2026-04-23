#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from anatomy_detection import config
from anatomy_detection.model import build_maskrcnn
from anatomy_tracking.io import (
    DETECTION_FIELDNAMES,
    frame_id_from_name,
    label_from_meta,
    load_index,
    meta_by_video,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained anatomy Mask R-CNN and export tracker detections.csv.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default=config.DEFAULT_INDEX_JSON)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--output-dir", default=os.path.join(config.DEFAULT_OUTPUT_DIR, "predictions"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections-per-frame", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frame-source", default="all", choices=["all", "annotated"])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def split_video_ids(splits_path: Optional[str], fold: Optional[int], split: Optional[str]) -> Optional[set]:
    if not splits_path or fold is None or not split or split == "all":
        return None
    import json

    with open(splits_path, "r") as f:
        splits = json.load(f)
    return {str(video_id) for video_id in splits[f"fold_{int(fold)}"][split]}


def selected_video_ids(index: Sequence[Dict], args: argparse.Namespace) -> List[str]:
    ids = [str(item["video_id"]) for item in index]
    split_ids = split_video_ids(args.splits, args.fold, args.split)
    explicit_ids = {str(video_id) for video_id in args.video_ids} if args.video_ids else None
    selected = []
    for video_id in ids:
        if split_ids is not None and video_id not in split_ids:
            continue
        if explicit_ids is not None and video_id not in explicit_ids:
            continue
        selected.append(video_id)
    return selected


def frame_names_for_meta(meta: Dict, source: str) -> List[str]:
    if source == "annotated":
        frames = list(meta.get("annotated_frame_names") or [])
        if frames:
            return frames
    return list(meta.get("frame_names") or meta.get("frames_names") or [])


def iter_frames(meta: Dict, frame_source: str, frame_stride: int) -> Iterable[Dict]:
    frames_dir = str(meta.get("frames_dir", ""))
    frame_names = frame_names_for_meta(meta, frame_source)
    if frame_stride > 1:
        frame_names = frame_names[:: int(frame_stride)]
    for frame_name in frame_names:
        image_path = os.path.join(frames_dir, frame_name)
        if not os.path.exists(image_path):
            continue
        yield {
            "video_id": str(meta["video_id"]),
            "true_label": label_from_meta(meta),
            "frame_id": frame_id_from_name(frame_name),
            "frame_name": frame_name,
            "image_path": image_path,
        }


def mask_centroid_and_area(mask: np.ndarray, fallback_box: Sequence[float]) -> tuple:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        x1, y1, x2, y2 = fallback_box
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    return float(np.mean(xs)), float(np.mean(ys)), float(len(xs))


def output_rows_for_frame(
    frame: Dict,
    output: Dict,
    id_to_class: Dict[int, str],
    image_width: int,
    image_height: int,
    score_threshold: float,
    mask_threshold: float,
    max_detections: int,
) -> List[Dict]:
    scores = output.get("scores", torch.empty(0)).detach().cpu().numpy()
    labels = output.get("labels", torch.empty(0)).detach().cpu().numpy()
    boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
    masks = output.get("masks")
    if masks is not None:
        masks_np = masks.detach().cpu().numpy()[:, 0, :, :]
    else:
        masks_np = np.zeros((len(scores), int(image_height), int(image_width)), dtype=np.float32)

    order = np.argsort(-scores)
    rows = []
    for idx in order[:max_detections]:
        score = float(scores[idx])
        if score < score_threshold:
            continue
        class_id = int(labels[idx])
        class_name = id_to_class.get(class_id)
        if class_name is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in boxes[idx]]
        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))
        if x2 <= x1 or y2 <= y1:
            continue

        mask = masks_np[idx] >= float(mask_threshold)
        centroid_x, centroid_y, area = mask_centroid_and_area(mask, (x1, y1, x2, y2))
        rows.append(
            {
                "video_id": frame["video_id"],
                "true_label": int(frame["true_label"]),
                "frame_id": int(frame["frame_id"]),
                "frame_name": frame["frame_name"],
                "class_name": class_name,
                "original_label": class_name,
                "confidence": score,
                "shape_type": "mask",
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "area": area,
                "image_width": int(image_width),
                "image_height": int(image_height),
                "ann_path": "",
                "image_path": os.path.abspath(frame["image_path"]),
                "points_json": "[]",
            }
        )
    return rows


def batched(items: Sequence[Dict], batch_size: int) -> Iterable[Sequence[Dict]]:
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    target_classes = list(checkpoint.get("target_classes") or config.TARGET_CLASSES)
    id_to_class = {idx + 1: class_name for idx, class_name in enumerate(target_classes)}
    saved_args = checkpoint.get("args", {})
    min_size = int(args.min_size or saved_args.get("min_size") or 640)
    max_size = int(args.max_size or saved_args.get("max_size") or 1024)

    device = pick_device(args.device)
    model = build_maskrcnn(
        num_classes=len(target_classes) + 1,
        pretrained=False,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=saved_args.get("trainable_backbone_layers", 3),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    index = load_index(args.index)
    meta_map = meta_by_video(index)
    video_ids = selected_video_ids(index, args)
    rows: List[Dict] = []

    print(f"Running detector on {len(video_ids)} videos with device={device}", flush=True)
    with torch.no_grad():
        for video_idx, video_id in enumerate(video_ids, start=1):
            frames = list(iter_frames(meta_map[video_id], args.frame_source, args.frame_stride))
            total_frames = len(frames)
            frame_count = 0
            print(f"{video_idx}/{len(video_ids)} {video_id}: starting {total_frames} frames", flush=True)
            for frame_batch in batched(frames, args.batch_size):
                tensors = []
                frame_infos = []
                for frame in frame_batch:
                    image = Image.open(frame["image_path"]).convert("RGB")
                    width, height = image.size
                    tensors.append(image_tensor(image).to(device))
                    frame_infos.append((frame, width, height))
                outputs = model(tensors)
                for (frame, width, height), output in zip(frame_infos, outputs):
                    rows.extend(
                        output_rows_for_frame(
                            frame=frame,
                            output=output,
                            id_to_class=id_to_class,
                            image_width=width,
                            image_height=height,
                            score_threshold=args.score_threshold,
                            mask_threshold=args.mask_threshold,
                            max_detections=args.max_detections_per_frame,
                        )
                    )
                    frame_count += 1
                    if args.progress_every > 0 and frame_count % args.progress_every == 0:
                        print(
                            f"{video_idx}/{len(video_ids)} {video_id}: processed {frame_count}/{total_frames} frames",
                            flush=True,
                        )
            print(f"{video_idx}/{len(video_ids)} {video_id}: processed {frame_count} frames", flush=True)

    output_csv = os.path.join(args.output_dir, "detections.csv")
    write_csv(output_csv, rows, DETECTION_FIELDNAMES)
    print(f"Wrote {len(rows)} predicted anatomy detections to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
