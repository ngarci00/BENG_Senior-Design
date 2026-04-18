from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from anatomy_tracking.io import numeric

#Compute intersection over union (IoU) between two bounding boxes, which is a common metric for evaluating object detection performance by measuring the overlap between predicted and ground truth boxes.
def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0

#Compute norm centroid distance between two detections.
def normalized_centroid_distance(a: Dict, b: Dict) -> float:
    width = max(numeric(a, "image_width", 0.0), numeric(b, "image_width", 0.0), 1.0)
    height = max(numeric(a, "image_height", 0.0), numeric(b, "image_height", 0.0), 1.0)
    dx = (numeric(a, "centroid_x") - numeric(b, "centroid_x")) / width
    dy = (numeric(a, "centroid_y") - numeric(b, "centroid_y")) / height
    return float(np.sqrt(dx * dx + dy * dy))

#Compute bounding box from detection row, which extracts the bounding box coordinates and dimensions from the row data for use in tracking and evaluation.
def detection_bbox(row: Dict) -> Tuple[float, float, float, float]:
    return (
        numeric(row, "bbox_x"),
        numeric(row, "bbox_y"),
        numeric(row, "bbox_w"),
        numeric(row, "bbox_h"),
    )
#sorts detection rows by video_id, class_name, frame_id, confidence, and bounding box position to ensure a consistent order for processing and evaluation.
def sort_detection_rows(rows: Iterable[Dict]) -> List[Dict]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("video_id", "")),
            str(row.get("class_name", "")),
            int(numeric(row, "frame_id", -1)),
            -numeric(row, "confidence", 1.0),
            numeric(row, "bbox_x"),
            numeric(row, "bbox_y"),
        ),
    )

#Link detections across video frames into tracks based on spatial and temporal proximity, using IoU and centroid distance as criteria for associating detections into the same track, while allowing for gaps in detection.
def link_video_class_detections(
    rows: List[Dict],
    track_prefix: str,
    min_iou: float = 0.1,
    max_center_distance: float = 0.08,
    max_frame_gap: int = 5,
) -> List[Dict]:
    rows = sorted(rows, key=lambda row: (int(numeric(row, "frame_id", -1)), -numeric(row, "confidence", 1.0)))
    active: List[Dict] = []
    next_track_idx = 1
    assigned: List[Dict] = []

    for row in rows:
        frame_id = int(numeric(row, "frame_id", -1))
        best_track: Optional[Dict] = None
        best_score = -1.0

        for track in active:
            gap = frame_id - int(track["last_frame_id"])
            if gap < 0 or gap > max_frame_gap:
                continue
            iou = bbox_iou(detection_bbox(track["last_row"]), detection_bbox(row))
            distance = normalized_centroid_distance(track["last_row"], row)
            if iou < min_iou and distance > max_center_distance:
                continue
            score = iou - distance - 0.01 * max(gap - 1, 0)
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None:
            track_id = f"{track_prefix}_{next_track_idx:03d}"
            next_track_idx += 1
            best_track = {
                "track_id": track_id,
                "last_frame_id": frame_id,
                "last_row": row,
                "length": 0,
            }
            active.append(best_track)
        else:
            best_track["last_frame_id"] = frame_id
            best_track["last_row"] = row

        best_track["length"] += 1
        output_row = dict(row)
        output_row["track_id"] = best_track["track_id"]
        output_row["track_local_index"] = int(best_track["length"])
        output_row["match_score"] = float(best_score) if best_score >= -0.5 else 0.0
        assigned.append(output_row)

        active = [
            track
            for track in active
            if frame_id - int(track["last_frame_id"]) <= max_frame_gap
        ]

    return assigned