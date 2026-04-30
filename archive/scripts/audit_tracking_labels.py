#!/usr/bin/env python3
"""Audit LabelMe anatomy annotations for tracker readiness.

Outputs:
  video_summary.csv       per-video coverage and missingness
  audit_summary.json      aggregate counts and warnings

Optional debug output:
  label_summary.csv       counts by canonical label and shape type
  original_label_summary.csv counts by raw LabelMe label
"""

import argparse
import collections
import json
import os
import sys
from typing import Dict, List

from _paths import add_archive_src_to_path

add_archive_src_to_path()

from anatomy_tracking import config
from anatomy_tracking.io import (
    annotation_paths_for_video,
    canonical_label,
    frame_id_from_name,
    iter_video_meta,
    load_index,
    read_json,
    write_csv,
    write_json,
)

#Parse command-line arguments for the audit script, allowing specification of input index, output directory, target classes, and options for writing detailed label summaries.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=config.DEFAULT_INDEX_JSON, help="Path to index_poly.json or index_rec.json")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.DEFAULT_OUTPUT_DIR, "anatomy_label_audit"),
        help="Directory for audit CSV/JSON files",
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=config.TARGET_CLASSES,
        help="Canonical labels expected by the tracker",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--write-label-details", action="store_true", help="Also write label detail CSVs")
    return parser.parse_args()

#Create an empty video row with default values for all expected fields, which serves as a template for populating video-level annotation statistics during the audit process.
def empty_video_row(meta: Dict, target_classes: List[str]) -> Dict:
    row = {
        "video_id": str(meta["video_id"]),
        "true_label": int(meta.get("label", 0)),
        "ann_dir": meta.get("ann_dir", ""),
        "frames_dir": meta.get("frames_dir", ""),
        "n_frames_index": len(meta.get("frame_names", []) or []),
        "n_annotation_files": 0,
        "n_frames_with_target": 0,
        "n_shapes_total": 0,
        "n_target_shapes": 0,
        "n_unknown_shapes": 0,
        "first_annotated_frame": "",
        "last_annotated_frame": "",
        "missing_annotation_files": 0,
        "bad_json_files": 0,
        "missing_image_size_files": 0,
    }
    for class_name in target_classes:
        row[f"{class_name}_frames"] = 0
        row[f"{class_name}_shapes"] = 0
    return row

#Main function to perform the audit of LabelMe annotations, which iterates through videos and their annotations to collect statistics on label usage, annotation completeness, and potential issues, ultimately summarizing the findings in CSV and JSON output files.
def main() -> None:
    args = parse_args()
    # Normalize requested labels once so spelling variants are audited consistently.
    target_classes = [canonical_label(label) for label in args.target_classes]
    target_set = set(target_classes)

    index = load_index(args.index)
    label_counter = collections.Counter()
    original_label_counter = collections.Counter()
    shape_type_counter = collections.Counter()
    unknown_counter = collections.Counter()
    video_rows: List[Dict] = []
    bad_files: List[Dict] = []

    for meta in iter_video_meta(index, max_videos=args.max_videos):
        row = empty_video_row(meta, target_classes)
        ann_paths = annotation_paths_for_video(meta)
        expected_frames = meta.get("annotated_frame_names") or []
        # Missing annotation files are useful for deciding whether gaps are real or just unlabeled.
        if expected_frames:
            row["missing_annotation_files"] = max(0, len(expected_frames) - len(ann_paths))

        frame_ids = []
        for ann_path in ann_paths:
            row["n_annotation_files"] += 1
            frame_ids.append(frame_id_from_name(os.path.basename(ann_path)))
            try:
                ann = read_json(ann_path)
            except Exception as exc:
                row["bad_json_files"] += 1
                bad_files.append({"video_id": meta["video_id"], "ann_path": ann_path, "error": str(exc)})
                continue

            if not ann.get("imageWidth") or not ann.get("imageHeight"):
                row["missing_image_size_files"] += 1

            labels_in_frame = set()
            for shape in ann.get("shapes", []):
                original_label = str(shape.get("label", ""))
                class_name = canonical_label(original_label)
                shape_type = str(shape.get("shape_type") or "")
                label_counter[(class_name, shape_type)] += 1
                original_label_counter[(original_label, class_name)] += 1
                shape_type_counter[shape_type] += 1
                row["n_shapes_total"] += 1

                if class_name in target_set:
                    row["n_target_shapes"] += 1
                    row[f"{class_name}_shapes"] += 1
                    labels_in_frame.add(class_name)
                else:
                    # Non-target labels are retained here so annotation drift is visible.
                    row["n_unknown_shapes"] += 1
                    unknown_counter[original_label] += 1

            if labels_in_frame:
                row["n_frames_with_target"] += 1
            for class_name in labels_in_frame:
                row[f"{class_name}_frames"] += 1

        valid_frame_ids = [frame_id for frame_id in frame_ids if frame_id >= 0]
        if valid_frame_ids:
            row["first_annotated_frame"] = min(valid_frame_ids)
            row["last_annotated_frame"] = max(valid_frame_ids)
        video_rows.append(row)

    label_rows = []
    for (class_name, shape_type), count in sorted(label_counter.items()):
        label_rows.append(
            {
                "class_name": class_name,
                "shape_type": shape_type,
                "count": int(count),
                "is_target": int(class_name in target_set),
            }
        )

    original_rows = []
    for (original_label, class_name), count in sorted(original_label_counter.items()):
        original_rows.append(
            {
                "original_label": original_label,
                "canonical_label": class_name,
                "count": int(count),
                "is_target": int(class_name in target_set),
            }
        )

    summary = {
        "index": os.path.abspath(args.index),
        "n_videos": len(video_rows),
        "target_classes": target_classes,
        "n_annotation_files": int(sum(row["n_annotation_files"] for row in video_rows)),
        "n_frames_with_target": int(sum(row["n_frames_with_target"] for row in video_rows)),
        "n_target_shapes": int(sum(row["n_target_shapes"] for row in video_rows)),
        "n_unknown_shapes": int(sum(row["n_unknown_shapes"] for row in video_rows)),
        "shape_types": dict(shape_type_counter),
        "unknown_labels": dict(unknown_counter),
        "bad_files": bad_files[:200],
        "notes": [
            "Canonical labels are normalized with anatomy_tracking.config.LABEL_SYNONYMS.",
            "Non-target labels are retained in the audit but omitted from tracker conversion by default.",
        ],
    }

    write_csv(os.path.join(args.output_dir, "video_summary.csv"), video_rows)
    write_json(os.path.join(args.output_dir, "audit_summary.json"), summary)
    if args.write_label_details:
        write_csv(os.path.join(args.output_dir, "label_summary.csv"), label_rows)
        write_csv(os.path.join(args.output_dir, "original_label_summary.csv"), original_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
