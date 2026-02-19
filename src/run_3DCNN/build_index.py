import os
import json

from config import raw_frames_dir, ann_dir, index_json_path, LABEL_TO_INT

IMG_EXTS = (".jpg", ".jpeg", ".png")
#Function builds an index of videos with their labels, directory, and frame names. 
def _resolve_frames_root() -> str:
    """Prefer config.raw_frames_dir; fall back to sibling raw_videos."""
    if os.path.isdir(raw_frames_dir):
        return raw_frames_dir
    #In case folder is not found:
    fallback = os.path.join(os.path.dirname(raw_frames_dir), "raw_videos")
    if os.path.isdir(fallback):
        print(f"raw_frames_dir not found; using {fallback}")
        return fallback
    #If neither folder exists, raise an error with instructions (:
    raise FileNotFoundError(
        "Frames root not found! Checked:\n"
        f"  - {raw_frames_dir}\n"
        f"  - {fallback}\n"
        "Fix config.py raw_frames_dir or create per-video frame folders."
    )

#frames are expected to be in per-vidoe folders:
def _sorted_frames(frames_dir: str):
    names = [f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS)]

    def k(name: str):
        stem = os.path.splitext(name)[0]
        try:
            return (0, int(stem))
        except Exception:
            return (1, stem)

    names.sort(key=k)
    return names

#For each video we gather the parameters below, and return it as a list of dicts to be written to the index json file: 
def _index_video(video_id: str, label_str: str, video_ann_dir: str, frames_dir: str):
    frame_names = _sorted_frames(frames_dir)#Only including frames that have annotations!

    ann_files = [f for f in os.listdir(video_ann_dir) if f.lower().endswith(".json")]
    ann_stems = set(os.path.splitext(f)[0] for f in ann_files)
    annotated_frame_names = [fn for fn in frame_names if os.path.splitext(fn)[0] in ann_stems]

    return {
        "video_id": video_id,
        "label": int(LABEL_TO_INT[label_str]),
        "frames_dir": os.path.abspath(frames_dir),
        "ann_dir": os.path.abspath(video_ann_dir),
        "frame_names": frame_names,
        "annotated_frame_names": annotated_frame_names,
    }

def main():
    frames_root = _resolve_frames_root()

    items = []

    for label_str in ("PASS", "FAIL"):
        label_dir = os.path.join(ann_dir, label_str)
        if not os.path.isdir(label_dir):
            continue

        for video_id in os.listdir(label_dir):
            if str(video_id).startswith("."):
                continue

            video_ann_dir = os.path.join(label_dir, video_id)
            if not os.path.isdir(video_ann_dir):
                continue

            frames_dir = os.path.join(frames_root, video_id)
            if not os.path.isdir(frames_dir):
                raise FileNotFoundError(
                    f"Missing frames for {video_id}: {frames_dir}\n"
                    "Expected per-video frame folders (e.g., data/videos/raw_videos/<video_id>/00000001.png)."
                )

            items.append(_index_video(video_id, label_str, video_ann_dir, frames_dir))

    if not items:
        raise RuntimeError(
            "No videos indexed! Verify:\n"
            f"  - annotations: {os.path.join(ann_dir, 'PASS')} and {os.path.join(ann_dir, 'FAIL')}\n"
            f"  - frames: {frames_root}/<video_id>/"
        )

    n_pass = sum(int(x["label"]) == 1 for x in items)
    n_fail = sum(int(x["label"]) == 0 for x in items)
    print(f"\nSummary: PASS={n_pass} FAIL={n_fail} Total={len(items)}")

    os.makedirs(os.path.dirname(index_json_path) or ".", exist_ok=True)
    with open(index_json_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"Saved index file to {index_json_path} !!!")

if __name__ == "__main__":
    main()