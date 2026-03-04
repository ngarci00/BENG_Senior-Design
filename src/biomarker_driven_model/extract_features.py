import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Ensure `<repo_root>/src` is on sys.path so `biomarker_driven_model` can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from biomarker_driven_model import config


_LABEL_SYNONYMS = {
    "vocalcords": "vocal_cords",
    "vocal cords": "vocal_cords",
    "vocal_cords": "vocal_cords",
    "cords": "vocal_cords",
    "glottis": "vocal_cords",
    "epiglottis": "epiglottis",
    "arytenoid": "arytenoids",
    "arytenoids": "arytenoids",
    "esophagus": "esophagus",
    "oesophagus": "esophagus",
    "ett": "endotracheal_tube",
    "endotracheal tube": "endotracheal_tube",
    "endotracheal_tube": "endotracheal_tube",
    "tube": "endotracheal_tube",
}


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _canon_label(label: str) -> str:
    key = str(label).strip().lower()
    return _LABEL_SYNONYMS.get(key, key)


def _polygon_area(points: Iterable[Iterable[float]]) -> float:
    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_video_features(meta: Dict) -> np.ndarray:
    ann_dir = meta.get("ann_dir", "")
    frames = list(meta.get("annotated_frame_names", []))

    frame_presence = {b: 0 for b in config.biomarkers}
    shape_counts = {b: 0 for b in config.biomarkers}
    areas = {b: [] for b in config.biomarkers}

    total_frames = 0
    for frame in frames:
        json_name = os.path.splitext(frame)[0] + ".json"
        path = os.path.join(ann_dir, json_name)
        if not os.path.exists(path):
            continue

        total_frames += 1
        ann = _read_json(path)
        labels_in_frame = set()

        for shape in ann.get("shapes", []):
            label = _canon_label(shape.get("label", ""))
            if label not in config.biomarkers:
                continue

            labels_in_frame.add(label)
            shape_counts[label] += 1
            areas[label].append(_polygon_area(shape.get("points", [])))

        for label in labels_in_frame:
            frame_presence[label] += 1

    features: List[float] = []
    denom = max(total_frames, 1)

    for biomarker in config.biomarkers:
        presence_rate = frame_presence[biomarker] / float(denom)
        mean_area = float(np.mean(areas[biomarker])) if areas[biomarker] else 0.0
        mean_count_per_frame = shape_counts[biomarker] / float(denom)
        features.extend([presence_rate, mean_area, mean_count_per_frame])

    return np.asarray(features, dtype=np.float32)


def _build_matrix(video_ids: List[str], meta_by_vid: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    kept_ids: List[str] = []

    for video_id in video_ids:
        meta = meta_by_vid.get(video_id)
        if meta is None:
            raise KeyError(f"Video id {video_id} not found in index file")
        rows.append(extract_video_features(meta))
        labels.append(int(meta["label"]))
        kept_ids.append(str(video_id))

    X = np.stack(rows, axis=0) if rows else np.empty((0, len(config.biomarkers) * 3), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    vids = np.asarray(kept_ids, dtype=object)
    return X, y, vids


def _cached_features_match_split(path: str, train_ids: List[str], val_ids: List[str]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return False

    required = {"vids_train", "vids_val"}
    if not required.issubset(set(data.files)):
        return False

    cached_train = [str(v) for v in data["vids_train"].tolist()]
    cached_val = [str(v) for v in data["vids_val"].tolist()]
    return cached_train == [str(v) for v in train_ids] and cached_val == [str(v) for v in val_ids]


def extract_fold(fold: int) -> None:
    fold = int(fold)
    _ensure_dir(config.features_dir)

    output_path = os.path.join(config.features_dir, f"fold_{fold}.npz")
    index = _read_json(config.index_json_path)
    splits = _read_json(config.splits_json_path)
    meta_by_vid = {str(item["video_id"]): item for item in index}

    split_key = f"fold_{fold}"
    if split_key not in splits:
        raise KeyError(f"Missing {split_key} in splits file: {config.splits_json_path}")

    train_ids = [str(v) for v in splits[split_key]["train"]]
    val_ids = [str(v) for v in splits[split_key]["val"]]

    if _cached_features_match_split(output_path, train_ids, val_ids):
        print(f"Features for fold {fold} already exist at {output_path}, skipping extraction.")
        return
    if os.path.exists(output_path):
        print(f"Features for fold {fold} exist but do not match the current split. Re-extracting.")

    X_train, y_train, vids_train = _build_matrix(train_ids, meta_by_vid)
    X_val, y_val, vids_val = _build_matrix(val_ids, meta_by_vid)

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        vids_train=vids_train,
        X_val=X_val,
        y_val=y_val,
        vids_val=vids_val,
    )
    print(
        f"Saved fold {fold} features to {output_path} "
        f"(train={len(vids_train)} videos, val={len(vids_val)} videos)"
    )


def main() -> None:
    for fold in range(int(config.kfolds)):
        extract_fold(fold)


if __name__ == "__main__":
    main()
