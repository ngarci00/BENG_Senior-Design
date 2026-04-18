#!/usr/bin/env python3
import argparse
import collections
import os
import sys
from typing import Dict, List
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
# Use the repo-local package directly from a script entry point.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from anatomy_tracking import config
from anatomy_tracking.io import ensure_dir, numeric, read_csv, write_csv
from anatomy_tracking.tracking import link_video_class_detections, sort_detection_rows

#Fieldnames for tracks.csv, captures key info about each tracked detection. 
TRACK_FIELDNAMES = [
    "video_id",
    "true_label",
    "frame_id",
    "frame_name",
    "class_name",
    "track_id",
    "track_local_index",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "centroid_x",
    "centroid_y",
    "area",
    "image_width",
    "image_height",
    "match_score",
]

#function to parse command line arguments for the anatomy tracking script, with defaults and help messages for each argument.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections-csv",
        default=os.path.join(config.TRACKING_FORMAT_DIR, "detections.csv"),
        help="Converted LabelMe detections.csv",
    )
    parser.add_argument("--output-dir", default=config.TRACKS_DIR, help="Directory for tracks.csv")
    parser.add_argument("--write-summary", action="store_true", help="Also write a debug tracking_summary.json")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--max-center-distance", type=float, default=0.08)
    parser.add_argument("--max-frame-gap", type=int, default=5)
    return parser.parse_args()

#keeps only columns needed for feature extraction & feedback generation
def compact_rows(rows: List[Dict]) -> List[Dict]:
    #Keep only the columns needed for feature extraction and feedback generation.
    return [{field: row.get(field, "") for field in TRACK_FIELDNAMES} for row in rows]

#Summarizez tracking results by video.
def summarize_tracks(rows: List[Dict]) -> Dict:
    by_video = collections.defaultdict(list)
    by_track = collections.defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
        by_track[row["track_id"]].append(row)

    videos = {}
    for video_id, video_rows in sorted(by_video.items()):
        tracks = collections.Counter(row["track_id"] for row in video_rows)
        classes = collections.Counter(row["class_name"] for row in video_rows)
        videos[video_id] = {
            "n_detections": len(video_rows),
            "n_tracks": len(tracks),
            "class_detection_counts": dict(classes),
        }

    track_rows = []
    for track_id, track_rows_for_id in sorted(by_track.items()):
        frame_ids = sorted(int(numeric(row, "frame_id", -1)) for row in track_rows_for_id)
        # Gaps capture disappearance/reappearance events for interpretability.
        gaps = [
            frame_ids[idx] - frame_ids[idx - 1]
            for idx in range(1, len(frame_ids))
            if frame_ids[idx] - frame_ids[idx - 1] > 1
        ]
        first = track_rows_for_id[0]
        track_rows.append(
            {
                "track_id": track_id,
                "video_id": first["video_id"],
                "class_name": first["class_name"],
                "n_detections": len(track_rows_for_id),
                "first_frame": min(frame_ids) if frame_ids else -1,
                "last_frame": max(frame_ids) if frame_ids else -1,
                "n_gaps": len(gaps),
                "max_gap": max(gaps) if gaps else 0,
            }
        )

    return {"videos": videos, "tracks": track_rows}

#Main function to orchestrate the entire pipeline, running each step in sequence with appropriate arguments and handling output directories and file paths.
def main() -> None:
    args = parse_args()
    # Baseline tracker consumes annotation-derived detector-style rows.
    input_rows = [
        row
        for row in read_csv(args.detections_csv)
        if numeric(row, "confidence", 1.0) >= args.min_confidence
    ]

    grouped = collections.defaultdict(list)
    for row in sort_detection_rows(input_rows):
        # Track each anatomy class independently inside each video.
        grouped[(row["video_id"], row["class_name"])].append(row)

    output_rows: List[Dict] = []
    for (video_id, class_name), rows in sorted(grouped.items()):
        track_prefix = f"{video_id}_{class_name}".replace(".", "_")
        output_rows.extend(
            link_video_class_detections(
                rows,
                track_prefix=track_prefix,
                min_iou=args.min_iou,
                max_center_distance=args.max_center_distance,
                max_frame_gap=args.max_frame_gap,
            )
        )

    ensure_dir(args.output_dir)
    output_rows = compact_rows(output_rows)
    write_csv(os.path.join(args.output_dir, "tracks.csv"), output_rows, TRACK_FIELDNAMES)
    if args.write_summary:
        from anatomy_tracking.io import write_json

        write_json(os.path.join(args.output_dir, "tracking_summary.json"), summarize_tracks(output_rows))
    print(f"Wrote {len(output_rows)} tracked detections to {os.path.join(args.output_dir, 'tracks.csv')}")


if __name__ == "__main__":
    main()
