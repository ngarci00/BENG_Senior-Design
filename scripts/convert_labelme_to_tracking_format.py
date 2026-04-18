#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
# Use repo-local anatomy_tracking helpers without requiring pip installation.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from anatomy_tracking import config
from anatomy_tracking.io import (
    DETECTION_FIELDNAMES,
    annotation_paths_for_video,
    canonical_label,
    iter_video_meta,
    load_index,
    parse_labelme_file,
    write_csv,
    write_json,
)

#Convert LabelMe annotations to a normalized tracking format (detections.csv) and optionally COCO JSON.
#COCO JSON is a common format for object detection/segmentation datasets, and can be used for training Mask R-CNN or similar models.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=config.DEFAULT_INDEX_JSON, help="Path to index_poly.json or index_rec.json")
    parser.add_argument(
        "--output-dir",
        default=config.TRACKING_FORMAT_DIR,
        help="Directory for converted detection/tracking files",
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=config.TARGET_CLASSES,
        help="Canonical labels to keep",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--write-coco", action="store_true", help="Also write COCO JSON and label_map.json")
    parser.add_argument("--write-summary", action="store_true", help="Also write conversion_summary.json")
    return parser.parse_args()

#COCO annotation format reference: https://cocodataset.org/#format-data
def row_to_coco_annotation(row: Dict, annotation_id: int, image_id: int, category_id: int) -> Dict:
    points = json.loads(row.get("points_json") or "[]")
    segmentation = []
    #Detector training only needs boxes, but preserving polygons enables mask/segmentation later.
    if points and row.get("shape_type") == "polygon":
        segmentation = [[float(coord) for point in points for coord in point[:2]]]
    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": [
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        ],
        "area": float(row["area"]),
        "iscrowd": 0,
        "segmentation": segmentation,
    }

#Build COCO JSON structure from normalized rows. COCO format stores image metadata separately and links annotations by image_id.
def build_coco(rows: List[Dict], label_map: Dict[str, int]) -> Dict:
    image_key_to_id: Dict[Tuple[str, int], int] = {}
    images: List[Dict] = []
    annotations: List[Dict] = []

    for row in rows:
        key = (str(row["video_id"]), int(row["frame_id"]))
        # COCO stores image metadata once, then links all annotations by image_id.
        if key not in image_key_to_id:
            image_key_to_id[key] = len(image_key_to_id) + 1
            images.append(
                {
                    "id": image_key_to_id[key],
                    "file_name": row["image_path"],
                    "width": int(row["image_width"]),
                    "height": int(row["image_height"]),
                    "video_id": row["video_id"],
                    "frame_id": int(row["frame_id"]),
                    "frame_name": row["frame_name"],
                }
            )
        annotations.append(
            row_to_coco_annotation(
                row=row,
                annotation_id=len(annotations) + 1,
                image_id=image_key_to_id[key],
                category_id=label_map[row["class_name"]],
            )
        )

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": int(category_id), "name": class_name}
            for class_name, category_id in sorted(label_map.items(), key=lambda item: item[1])
        ],
    }

#Main function iterates over videos and annotations, converts to normalized rows, and optionally builds COCO JSON. Errors are collected for summary reporting but don't stop the conversion process.
def main() -> None:
    args = parse_args()
    # Fixed category ids keep detector training and inference label maps stable.
    target_classes = [canonical_label(label) for label in args.target_classes]
    label_map = {class_name: idx + 1 for idx, class_name in enumerate(target_classes)}

    index = load_index(args.index)
    rows: List[Dict] = []
    parse_errors: List[Dict] = []

    for meta in iter_video_meta(index, max_videos=args.max_videos):
        for ann_path in annotation_paths_for_video(meta):
            records, error = parse_labelme_file(meta, ann_path, target_classes=target_classes, confidence=1.0)
            if error:
                parse_errors.append({"video_id": str(meta["video_id"]), "ann_path": ann_path, "error": error})
                continue
            rows.extend(record.to_row() for record in records)

    write_csv(os.path.join(args.output_dir, "detections.csv"), rows, DETECTION_FIELDNAMES)

    if args.write_summary:
        write_json(
            os.path.join(args.output_dir, "conversion_summary.json"),
            {
                "index": os.path.abspath(args.index),
                "n_records": len(rows),
                "n_videos": len(set(row["video_id"] for row in rows)),
                "target_classes": target_classes,
                "parse_errors": parse_errors[:200],
            },
        )

    if args.write_coco:
        write_json(os.path.join(args.output_dir, "label_map.json"), label_map)
        write_json(os.path.join(args.output_dir, "coco_instances.json"), build_coco(rows, label_map))

    print(f"Wrote {len(rows)} normalized detections to {os.path.join(args.output_dir, 'detections.csv')}")
    if parse_errors:
        print(f"Encountered {len(parse_errors)} parse errors; rerun with --write-summary for details")


if __name__ == "__main__":
    main()
