import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from anatomy_tracking import config

#helper function: ensure src directory is on the Python path for imports when running scripts directly.
def ensure_src_on_path() -> None:
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

#utility function for reading json files and csv files. 
def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

#utility function for writing json files and csv files. 
def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def write_csv(path: str, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    fieldnames = list(fieldnames or (rows[0].keys() if rows else []))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

#utility function for reading json files and csv files. 
def read_csv(path: str) -> List[Dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

#converts various label formats to a canonical form based on predefined synonyms, which helps standardize class labels across different annotators or datasets for consistent analysis and modeling.
def canonical_label(label: str) -> str:
    key = str(label or "").strip().lower().replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return config.LABEL_SYNONYMS.get(key, key.replace(" ", "_"))

#function to provide frame ids from frame names.
def frame_id_from_name(name: str) -> int:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    match = re.search(r"(\d+)$", stem)
    if not match:
        return -1
    return int(match.group(1))

def label_from_meta(meta: Dict) -> int:
    return int(meta.get("label", 1 if str(meta.get("split_label", "")).upper() == "PASS" else 0))

def label_name_from_meta(meta: Dict) -> str:
    return "PASS" if label_from_meta(meta) == 1 else "FAIL"

def load_index(path: str) -> List[Dict]:
    return read_json(path)

def meta_by_video(index: Sequence[Dict]) -> Dict[str, Dict]:
    return {str(item["video_id"]): item for item in index}

def iter_video_meta(index: Sequence[Dict], max_videos: Optional[int] = None) -> Iterable[Dict]:
    count = 0
    for meta in index:
        if max_videos is not None and count >= max_videos:
            break
        count += 1
        yield meta

def annotation_paths_for_video(meta: Dict) -> List[str]:
    ann_dir = meta.get("ann_dir", "")
    if not ann_dir or not os.path.isdir(ann_dir):
        return []

    frame_names = meta.get("annotated_frame_names") or []
    if frame_names:
        paths = []
        for frame_name in frame_names:
            json_name = os.path.splitext(os.path.basename(frame_name))[0] + ".json"
            path = os.path.join(ann_dir, json_name)
            if os.path.exists(path):
                paths.append(path)
        return sorted(paths)

    return sorted(
        os.path.join(ann_dir, name)
        for name in os.listdir(ann_dir)
        if name.lower().endswith(".json")
    )

def image_path_for_annotation(meta: Dict, ann: Dict, ann_path: str) -> str:
    image_path = ann.get("imagePath") or ""
    if os.path.isabs(image_path):
        return image_path

    candidates = []
    if image_path:
        candidates.append(os.path.normpath(os.path.join(os.path.dirname(ann_path), image_path)))
    frames_dir = meta.get("frames_dir", "")
    frame_name = os.path.splitext(os.path.basename(ann_path))[0] + ".jpg"
    if frames_dir:
        candidates.append(os.path.join(frames_dir, frame_name))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""

def polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def bbox_from_points(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return 0.0, 0.0, 0.0, 0.0
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 0]))
    y1 = float(np.max(pts[:, 1]))
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)

def centroid_from_points(points: Sequence[Sequence[float]], bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 2:
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    x, y, w, h = bbox
    return x + 0.5 * w, y + 0.5 * h

def shape_area(shape_type: str, points: Sequence[Sequence[float]], bbox: Tuple[float, float, float, float]) -> float:
    if str(shape_type).lower() == "polygon":
        return polygon_area(points)
    _x, _y, w, h = bbox
    return float(w * h)

@dataclass
class DetectionRecord:
    video_id: str
    true_label: int
    frame_id: int
    frame_name: str
    class_name: str
    original_label: str
    confidence: float
    shape_type: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    centroid_x: float
    centroid_y: float
    area: float
    image_width: int
    image_height: int
    ann_path: str
    image_path: str
    points_json: str

    def to_row(self) -> Dict:
        row = asdict(self)
        for key in [
            "confidence",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "centroid_x",
            "centroid_y",
            "area",
        ]:
            value = row[key]
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                row[key] = 0.0
        return row

def parse_labelme_file(
    meta: Dict,
    ann_path: str,
    target_classes: Optional[Sequence[str]] = None,
    confidence: float = 1.0,
) -> Tuple[List[DetectionRecord], Optional[str]]:
    try:
        ann = read_json(ann_path)
    except Exception as exc:
        return [], str(exc)

    target_set = set(target_classes or config.TARGET_CLASSES)
    frame_name = os.path.splitext(os.path.basename(ann_path))[0] + ".jpg"
    frame_id = frame_id_from_name(frame_name)
    image_width = int(ann.get("imageWidth") or 0)
    image_height = int(ann.get("imageHeight") or 0)
    image_path = image_path_for_annotation(meta, ann, ann_path)

    records: List[DetectionRecord] = []
    for shape in ann.get("shapes", []):
        original_label = str(shape.get("label", ""))
        class_name = canonical_label(original_label)
        if target_set and class_name not in target_set:
            continue

        points = shape.get("points") or []
        bbox = bbox_from_points(points)
        centroid = centroid_from_points(points, bbox)
        shape_type = str(shape.get("shape_type") or "")
        area = shape_area(shape_type, points, bbox)
        records.append(
            DetectionRecord(
                video_id=str(meta["video_id"]),
                true_label=label_from_meta(meta),
                frame_id=frame_id,
                frame_name=frame_name,
                class_name=class_name,
                original_label=original_label,
                confidence=float(confidence),
                shape_type=shape_type,
                bbox_x=float(bbox[0]),
                bbox_y=float(bbox[1]),
                bbox_w=float(bbox[2]),
                bbox_h=float(bbox[3]),
                centroid_x=float(centroid[0]),
                centroid_y=float(centroid[1]),
                area=float(area),
                image_width=image_width,
                image_height=image_height,
                ann_path=os.path.abspath(ann_path),
                image_path=os.path.abspath(image_path) if image_path else "",
                points_json=json.dumps(points, separators=(",", ":")),
            )
        )

    return records, None

DETECTION_FIELDNAMES = [
    "video_id",
    "true_label",
    "frame_id",
    "frame_name",
    "class_name",
    "original_label",
    "confidence",
    "shape_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "centroid_x",
    "centroid_y",
    "area",
    "image_width",
    "image_height",
    "ann_path",
    "image_path",
    "points_json",
]

def numeric(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except Exception:
        return default