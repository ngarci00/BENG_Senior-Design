#!/usr/bin/env python3
import argparse
import collections
import os
import sys
from typing import Dict, List, Sequence
import numpy as np
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
# Keep imports repo-local so this file can be run as a standalone script.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from anatomy_tracking import config
from anatomy_tracking.io import load_index, meta_by_video, numeric, read_csv, write_csv, write_json

BASE_COLUMNS = ["video_id", "true_label", "n_index_frames", "n_tracked_frames", "n_tracked_detections"]

#Categorical feedback columns for users based on key anatomy visibility and tube relationships.
FEEDBACK_COLUMNS = [
    "video_id",
    "true_label",
    "tracking_quality",
    "feedback_flag",
    "tube_visible_fraction",
    "vocal_cords_visible_fraction",
    "epiglottis_visible_fraction",
    "esophagus_visible_fraction",
    "tube_to_vocal_cords_near_fraction",
    "tube_to_vocal_cords_min_distance",
    "tube_to_esophagus_near_fraction",
    "tube_to_esophagus_min_distance",
    "tube_motion_max_step",
    "feedback_reasons",
]

#Parse command-line arguments for feature extraction, including input files, output directory, target classes, and thresholds.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-csv", default=os.path.join(config.TRACKS_DIR, "tracks.csv"))
    parser.add_argument("--index", default=config.DEFAULT_INDEX_JSON)
    parser.add_argument("--output-dir", default=config.FEATURES_DIR)
    parser.add_argument("--target-classes", nargs="*", default=config.TARGET_CLASSES)
    parser.add_argument("--near-threshold", type=float, default=0.12, help="Normalized centroid distance threshold")
    parser.add_argument("--write-schema", action="store_true", help="Also write feature_schema.json")
    return parser.parse_args()

#Calculate normalized centroid distance between two detections, accounting for frame size to ensure comparability across videos.
def norm_centroid_distance(a: Dict, b: Dict) -> float:
    # Normalize by frame size so distances are comparable across resolutions.
    width = max(numeric(a, "image_width", 0.0), numeric(b, "image_width", 0.0), 1.0)
    height = max(numeric(a, "image_height", 0.0), numeric(b, "image_height", 0.0), 1.0)
    dx = (numeric(a, "centroid_x") - numeric(b, "centroid_x")) / width
    dy = (numeric(a, "centroid_y") - numeric(b, "centroid_y")) / height
    return float(np.sqrt(dx * dx + dy * dy))

#Helper functions to compute features related to the presence, confidence, area, and relationships of tracked anatomical structures across video frames.
def class_rows(rows: Sequence[Dict], class_name: str) -> List[Dict]:
    return [row for row in rows if row.get("class_name") == class_name]

#Extract frame IDs with detections to compute presence fractions and track continuity.
def frames_with(rows: Sequence[Dict]) -> set:
    return {int(numeric(row, "frame_id", -1)) for row in rows if int(numeric(row, "frame_id", -1)) >= 0}

#Calculate the fraction of indexed frames in which a given class appears in a continuous track, providing a signal for how consistently the anatomy is visible and tracked.
def longest_track_fraction(rows: Sequence[Dict], denom: int) -> float:
    if not rows:
        return 0.0
    counts = collections.Counter(row.get("track_id", "") for row in rows)
    return max(counts.values()) / float(max(denom, 1))

#Count gaps in tracking for a class, which may indicate intermittent visibility or tracking failures that could impact feature reliability.
def track_gap_count(rows: Sequence[Dict]) -> int:
    by_track = collections.defaultdict(list)
    for row in rows:
        by_track[row.get("track_id", "")].append(int(numeric(row, "frame_id", -1)))
    gaps = 0
    for frame_ids in by_track.values():
        frame_ids = sorted(frame_id for frame_id in frame_ids if frame_id >= 0)
        gaps += sum(1 for idx in range(1, len(frame_ids)) if frame_ids[idx] - frame_ids[idx - 1] > 1)
    return gaps

#Extract a comprehensive set of features for a given video, including class-specific presence and confidence metrics, pairwise relationships between key anatomical structures, and tube motion characteristics that may signal potential intubation issues.
def per_class_features(rows: Sequence[Dict], class_name: str, denom: int) -> Dict[str, float]:
    selected = class_rows(rows, class_name)
    selected_frames = frames_with(selected)
    areas = []
    confidences = []
    xs = []
    ys = []
    for row in selected:
        width = max(numeric(row, "image_width", 0.0), 1.0)
        height = max(numeric(row, "image_height", 0.0), 1.0)
        areas.append(numeric(row, "area", 0.0) / (width * height))
        confidences.append(numeric(row, "confidence", 1.0))
        xs.append(numeric(row, "centroid_x", 0.0) / width)
        ys.append(numeric(row, "centroid_y", 0.0) / height)

    track_ids = {row.get("track_id", "") for row in selected if row.get("track_id", "")}
    return {
        f"{class_name}_presence_fraction": len(selected_frames) / float(max(denom, 1)),
        f"{class_name}_detection_count": float(len(selected)),
        f"{class_name}_track_count": float(len(track_ids)),
        f"{class_name}_longest_track_fraction": longest_track_fraction(selected, denom),
        f"{class_name}_mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        f"{class_name}_mean_area_fraction": float(np.mean(areas)) if areas else 0.0,
        f"{class_name}_max_area_fraction": float(np.max(areas)) if areas else 0.0,
        f"{class_name}_mean_centroid_x": float(np.mean(xs)) if xs else 0.0,
        f"{class_name}_mean_centroid_y": float(np.mean(ys)) if ys else 0.0,
        f"{class_name}_gap_count": float(track_gap_count(selected)),
    }

#When multiple detections of the same class exist in a frame, use the one with the highest confidence to represent that frame for pairwise distance features, which helps mitigate noise from spurious detections and focuses on the most likely anatomical localization.
def best_by_frame(rows: Sequence[Dict], class_name: str) -> Dict[int, Dict]:
    best = {}
    for row in class_rows(rows, class_name):
        frame_id = int(numeric(row, "frame_id", -1))
        if frame_id < 0:
            continue
        # Use the highest-confidence detection when multiple same-class shapes exist in one frame.
        if frame_id not in best or numeric(row, "confidence", 1.0) > numeric(best[frame_id], "confidence", 1.0):
            best[frame_id] = row
    return best

def pair_features(
    rows: Sequence[Dict],
    class_a: str,
    class_b: str,
    denom: int,
    near_threshold: float,
) -> Dict[str, float]:
    a_by_frame = best_by_frame(rows, class_a)
    b_by_frame = best_by_frame(rows, class_b)
    common_frames = sorted(set(a_by_frame).intersection(b_by_frame))
    distances = [norm_centroid_distance(a_by_frame[frame], b_by_frame[frame]) for frame in common_frames]
    # Near-frame features approximate whether the tube approaches key airway landmarks.
    near_frames = [frame for frame, distance in zip(common_frames, distances) if distance <= near_threshold]
    prefix = f"{class_a}_to_{class_b}"
    return {
        f"{prefix}_cooccurrence_fraction": len(common_frames) / float(max(denom, 1)),
        f"{prefix}_min_distance": float(np.min(distances)) if distances else 1.0,
        f"{prefix}_mean_distance": float(np.mean(distances)) if distances else 1.0,
        f"{prefix}_near_fraction": len(near_frames) / float(max(denom, 1)),
        f"{prefix}_first_near_frame_fraction": (min(near_frames) / float(max(denom - 1, 1))) if near_frames else 1.0,
    }

#Tube motion features capture the stability and trajectory of the endotracheal tube across frames, which can provide signals for potential intubation issues such as dislodgement or misalignment if large jumps or inconsistent tracking are detected:
def tube_motion_features(rows: Sequence[Dict], denom: int) -> Dict[str, float]:
    tube_by_frame = best_by_frame(rows, "endotracheal_tube")
    frame_ids = sorted(tube_by_frame)
    steps = []
    for prev_frame, curr_frame in zip(frame_ids[:-1], frame_ids[1:]):
        if curr_frame <= prev_frame:
            continue
        steps.append(norm_centroid_distance(tube_by_frame[prev_frame], tube_by_frame[curr_frame]) / (curr_frame - prev_frame))
    return {
        "endotracheal_tube_motion_mean_step": float(np.mean(steps)) if steps else 0.0,
        "endotracheal_tube_motion_std_step": float(np.std(steps)) if steps else 0.0,
        "endotracheal_tube_motion_max_step": float(np.max(steps)) if steps else 0.0,
        "endotracheal_tube_first_seen_fraction": (min(frame_ids) / float(max(denom - 1, 1))) if frame_ids else 1.0,
        "endotracheal_tube_last_seen_fraction": (max(frame_ids) / float(max(denom - 1, 1))) if frame_ids else 0.0,
    }

#Build a feature row for a given video by aggregating class-specific features and pairwise relationships, which can then be used for training or evaluating an anatomy classifier to predict intubation success based on the presence and configuration of key anatomical landmarks.
def build_feature_row(meta: Dict, rows: Sequence[Dict], target_classes: Sequence[str], near_threshold: float) -> Dict:
    n_index_frames = len(meta.get("frame_names", []) or [])
    if n_index_frames == 0:
        frame_ids = [int(numeric(row, "frame_id", -1)) for row in rows]
        n_index_frames = max(frame_ids) + 1 if frame_ids else 0
    denom = max(n_index_frames, 1)
    tracked_frames = frames_with(rows)
    output = {
        "video_id": str(meta["video_id"]),
        "true_label": int(meta.get("label", 0)),
        "n_index_frames": int(n_index_frames),
        "n_tracked_frames": int(len(tracked_frames)),
        "n_tracked_detections": int(len(rows)),
    }
    for class_name in target_classes:
        output.update(per_class_features(rows, class_name, denom))

    # These pairwise relationships are the core anatomy-aware signals for PASS/FAIL.
    output.update(pair_features(rows, "endotracheal_tube", "vocal_cords", denom, near_threshold))
    output.update(pair_features(rows, "endotracheal_tube", "epiglottis", denom, near_threshold))
    output.update(pair_features(rows, "endotracheal_tube", "esophagus", denom, near_threshold))
    output.update(pair_features(rows, "vocal_cords", "esophagus", denom, near_threshold))
    output.update(tube_motion_features(rows, denom))
    return output

#Determine the set of feature columns based on the base columns and any additional features extracted from the data, which allows for flexible inclusion of new features without hardcoding column names.
def feature_columns(rows: Sequence[Dict]) -> List[str]:
    if not rows:
        return BASE_COLUMNS
    keys = list(rows[0].keys())
    return BASE_COLUMNS + [key for key in keys if key not in BASE_COLUMNS]

def value(row: Dict, key: str) -> float:
    return float(row.get(key) or 0.0)

def pct(value_float: float) -> str:
    return f"{100.0 * value_float:.1f}%"

#Build compact feedback signals based on key anatomy visibility and tube relationships, which can be used to provide interpretable insights to users about potential reasons for intubation failure or uncertainty in the anatomy classifier's predictions:
def build_feedback_row(row: Dict) -> Dict:
    tube_visible = value(row, "endotracheal_tube_presence_fraction")
    vocal_visible = value(row, "vocal_cords_presence_fraction")
    epiglottis_visible = value(row, "epiglottis_presence_fraction")
    esophagus_visible = value(row, "esophagus_presence_fraction")
    tube_vocal_near = value(row, "endotracheal_tube_to_vocal_cords_near_fraction")
    tube_vocal_min = value(row, "endotracheal_tube_to_vocal_cords_min_distance")
    tube_esophagus_near = value(row, "endotracheal_tube_to_esophagus_near_fraction")
    tube_esophagus_min = value(row, "endotracheal_tube_to_esophagus_min_distance")
    tube_motion_max = value(row, "endotracheal_tube_motion_max_step")

    #array of clinician readable feedback reasons used to explain potential signals of intubation failure or uncertainty!
    reasons = []
    if tube_visible < 0.05:
        reasons.append("ET tube rarely visible")
    else:
        reasons.append(f"ET tube visible in {pct(tube_visible)} of indexed frames")

    if vocal_visible < 0.05:
        reasons.append("vocal cords rarely visible")
    elif tube_vocal_near > 0:
        reasons.append(f"tube near vocal cords in {pct(tube_vocal_near)} of indexed frames")
    else:
        reasons.append("vocal cords visible but tube does not clearly approach them")

    if epiglottis_visible >= 0.05:
        reasons.append(f"epiglottis visible in {pct(epiglottis_visible)} of indexed frames")

    if tube_esophagus_near > 0:
        reasons.append(f"tube near esophagus in {pct(tube_esophagus_near)} of indexed frames")
    elif esophagus_visible >= 0.10:
        reasons.append(f"esophagus visible in {pct(esophagus_visible)} of indexed frames")

    if tube_motion_max > 0.20:
        reasons.append("large tube trajectory jump detected")

    if tube_visible < 0.05 or (vocal_visible < 0.05 and epiglottis_visible < 0.05):
        feedback_flag = "limited_anatomy_evidence"
    elif tube_esophagus_near >= 0.01 and tube_esophagus_near > tube_vocal_near:
        feedback_flag = "esophagus_risk"
    elif tube_vocal_near >= 0.02:
        feedback_flag = "airway_alignment"
    elif tube_vocal_near > 0:
        feedback_flag = "possible_airway_alignment"
    else:
        feedback_flag = "uncertain_alignment"

    tracking_quality = min(1.0, max(tube_visible, vocal_visible, epiglottis_visible))

    return {
        "video_id": row["video_id"],
        "true_label": int(row["true_label"]),
        "tracking_quality": float(tracking_quality),
        "feedback_flag": feedback_flag,
        "tube_visible_fraction": float(tube_visible),
        "vocal_cords_visible_fraction": float(vocal_visible),
        "epiglottis_visible_fraction": float(epiglottis_visible),
        "esophagus_visible_fraction": float(esophagus_visible),
        "tube_to_vocal_cords_near_fraction": float(tube_vocal_near),
        "tube_to_vocal_cords_min_distance": float(tube_vocal_min),
        "tube_to_esophagus_near_fraction": float(tube_esophagus_near),
        "tube_to_esophagus_min_distance": float(tube_esophagus_min),
        "tube_motion_max_step": float(tube_motion_max),
        "feedback_reasons": " | ".join(reasons),
    }

#main function: feature extraction pipeline for each video, loads data, computes and writes feature and feedback tables. 
def main() -> None:
    args = parse_args()
    index = load_index(args.index)
    meta_map = meta_by_video(index)
    rows = read_csv(args.tracks_csv)
    rows_by_video = collections.defaultdict(list)
    for row in rows:
        rows_by_video[str(row["video_id"])].append(row)

    feature_rows = []
    for video_id, meta in sorted(meta_map.items()):
        feature_rows.append(
            build_feature_row(meta, rows_by_video.get(video_id, []), args.target_classes, args.near_threshold)
        )

    output_csv = os.path.join(args.output_dir, "anatomy_features.csv")
    write_csv(output_csv, feature_rows, feature_columns(feature_rows))
    feedback_rows = [build_feedback_row(row) for row in feature_rows]
    feedback_csv = os.path.join(args.output_dir, "anatomy_feedback.csv")
    write_csv(feedback_csv, feedback_rows, FEEDBACK_COLUMNS)
    if args.write_schema:
        write_json(
            os.path.join(args.output_dir, "feature_schema.json"),
            {
                "base_columns": BASE_COLUMNS,
                "feedback_columns": FEEDBACK_COLUMNS,
                "target_classes": args.target_classes,
                "near_threshold": args.near_threshold,
                "feature_columns": feature_columns(feature_rows),
            },
        )
    print(f"Wrote anatomy feature table for {len(feature_rows)} videos to {output_csv}")
    print(f"Wrote compact feedback signals to {feedback_csv}")

if __name__ == "__main__":
    main()
