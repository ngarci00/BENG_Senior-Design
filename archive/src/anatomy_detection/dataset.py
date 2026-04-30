import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from anatomy_detection import config
from anatomy_tracking.io import (
    annotation_paths_for_video,
    canonical_label,
    frame_id_from_name,
    iter_video_meta,
    load_index,
    parse_labelme_file,
)


@dataclass(frozen=True)
class FrameAnnotation:
    video_id: str
    true_label: int
    frame_id: int
    frame_name: str
    image_path: str
    ann_path: str


def collate_detection_batch(batch):
    return tuple(zip(*batch))


def target_classes_from_args(classes: Optional[Sequence[str]] = None) -> List[str]:
    return [canonical_label(class_name) for class_name in (classes or config.TARGET_CLASSES)]


def class_maps(target_classes: Optional[Sequence[str]] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = target_classes_from_args(target_classes)
    class_to_id = {class_name: idx + 1 for idx, class_name in enumerate(classes)}
    id_to_class = {idx: class_name for class_name, idx in class_to_id.items()}
    return class_to_id, id_to_class


def _split_video_ids(splits_path: Optional[str], fold: Optional[int], split: Optional[str]) -> Optional[set]:
    if splits_path is None or fold is None or split is None:
        return None
    with open(splits_path, "r") as f:
        splits = json.load(f)
    return {str(video_id) for video_id in splits[f"fold_{int(fold)}"][str(split)]}


def _limit_paths(paths: List[str], max_frames_per_video: Optional[int], seed: int) -> List[str]:
    if max_frames_per_video is None or len(paths) <= max_frames_per_video:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, int(max_frames_per_video)))


def build_frame_annotations(
    index_path: str,
    splits_path: Optional[str] = None,
    fold: Optional[int] = None,
    split: Optional[str] = None,
    video_ids: Optional[Sequence[str]] = None,
    target_classes: Optional[Sequence[str]] = None,
    max_frames_per_video: Optional[int] = None,
    frame_stride: int = 1,
    include_empty: bool = False,
    seed: int = 42,
) -> List[FrameAnnotation]:
    selected_ids = _split_video_ids(splits_path, fold, split)
    if video_ids is not None:
        explicit_ids = {str(video_id) for video_id in video_ids}
        selected_ids = explicit_ids if selected_ids is None else selected_ids.intersection(explicit_ids)

    target_classes = target_classes_from_args(target_classes)
    records: List[FrameAnnotation] = []
    for meta in iter_video_meta(load_index(index_path)):
        video_id = str(meta["video_id"])
        if selected_ids is not None and video_id not in selected_ids:
            continue

        ann_paths = annotation_paths_for_video(meta)
        if frame_stride > 1:
            ann_paths = ann_paths[:: int(frame_stride)]
        ann_paths = _limit_paths(ann_paths, max_frames_per_video, seed + len(records))

        for ann_path in ann_paths:
            detections, error = parse_labelme_file(meta, ann_path, target_classes=target_classes)
            if error:
                continue
            if not include_empty and not detections:
                continue
            image_path = detections[0].image_path if detections else ""
            if not image_path:
                image_name = os.path.splitext(os.path.basename(ann_path))[0] + ".jpg"
                image_path = os.path.join(str(meta.get("frames_dir", "")), image_name)
            if not image_path or not os.path.exists(image_path):
                continue
            frame_name = os.path.basename(image_path)
            records.append(
                FrameAnnotation(
                    video_id=video_id,
                    true_label=int(meta.get("label", 0)),
                    frame_id=frame_id_from_name(frame_name),
                    frame_name=frame_name,
                    image_path=image_path,
                    ann_path=ann_path,
                )
            )
    return records


def _polygon_mask(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (int(width), int(height)), 0)
    if len(points) >= 3:
        xy = [(float(point[0]), float(point[1])) for point in points if len(point) >= 2]
        if len(xy) >= 3:
            ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _box_mask(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = Image.new("L", (int(width), int(height)), 0)
    ImageDraw.Draw(mask).rectangle((float(x1), float(y1), float(x2), float(y2)), outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _horizontal_flip(image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
    width = image.shape[-1]
    image = torch.flip(image, dims=(-1,))
    boxes = target["boxes"].clone()
    if boxes.numel():
        old_x1 = boxes[:, 0].clone()
        old_x2 = boxes[:, 2].clone()
        boxes[:, 0] = width - old_x2
        boxes[:, 2] = width - old_x1
        target["boxes"] = boxes
    if target["masks"].numel():
        target["masks"] = torch.flip(target["masks"], dims=(-1,))
    return image, target


class LabelMeMaskDataset(torch.utils.data.Dataset):
    """Torchvision Mask R-CNN dataset backed by LabelMe polygon annotations."""

    def __init__(
        self,
        index_path: str = config.DEFAULT_INDEX_JSON,
        splits_path: Optional[str] = None,
        fold: Optional[int] = None,
        split: Optional[str] = None,
        video_ids: Optional[Sequence[str]] = None,
        target_classes: Optional[Sequence[str]] = None,
        max_frames_per_video: Optional[int] = None,
        frame_stride: int = 1,
        include_empty: bool = False,
        hflip_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.index_path = index_path
        self.target_classes = target_classes_from_args(target_classes)
        self.class_to_id, self.id_to_class = class_maps(self.target_classes)
        self.hflip_prob = float(hflip_prob)
        self.seed = int(seed)
        self.samples = build_frame_annotations(
            index_path=index_path,
            splits_path=splits_path,
            fold=fold,
            split=split,
            video_ids=video_ids,
            target_classes=self.target_classes,
            max_frames_per_video=max_frames_per_video,
            frame_stride=frame_stride,
            include_empty=include_empty,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_target(self, sample: FrameAnnotation, width: int, height: int) -> Dict:
        detections, error = parse_labelme_file(
            {"video_id": sample.video_id, "label": sample.true_label, "frames_dir": os.path.dirname(sample.image_path)},
            sample.ann_path,
            target_classes=self.target_classes,
        )
        if error:
            raise RuntimeError(f"Could not parse {sample.ann_path}: {error}")

        boxes = []
        labels = []
        masks = []
        areas = []
        for det in detections:
            x1 = max(0.0, min(float(det.bbox_x), float(width - 1)))
            y1 = max(0.0, min(float(det.bbox_y), float(height - 1)))
            x2 = max(0.0, min(float(det.bbox_x + det.bbox_w), float(width)))
            y2 = max(0.0, min(float(det.bbox_y + det.bbox_h), float(height)))
            if x2 <= x1 or y2 <= y1:
                continue

            points = json.loads(det.points_json or "[]")
            if str(det.shape_type).lower() == "polygon" and len(points) >= 3:
                mask = _polygon_mask(points, width, height)
            else:
                mask = _box_mask((x1, y1, x2, y2), width, height)
            mask_area = float(mask.sum())
            if mask_area <= 0.0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[det.class_name])
            masks.append(mask)
            areas.append(mask_area)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            if masks
            else torch.zeros((0, int(height), int(width)), dtype=torch.uint8),
            "image_id": torch.as_tensor([sample.frame_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "video_id": sample.video_id,
            "frame_id": sample.frame_id,
            "frame_name": sample.frame_name,
            "image_path": sample.image_path,
        }
        return target

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        width, height = image.size
        image_tensor = _to_image_tensor(image)
        target = self._load_target(sample, width=width, height=height)
        target["image_id"] = torch.as_tensor([idx], dtype=torch.int64)

        rng = random.Random(self.seed + idx)
        if self.hflip_prob > 0.0 and rng.random() < self.hflip_prob:
            image_tensor, target = _horizontal_flip(image_tensor, target)
        return image_tensor, target
