# The goal of this script is to ensure all frames have matching JSON files, before any training is done.
import json
from pathlib import Path

# Directories and file paths
raw_dir = Path("data/videos/raw_videos")  # JPG frames (per-video folders)
ann_dir = Path("data/videos/rectangle_label_videos")  # LabelMe JSONs (per-video folders)
out_json = Path("data/videos/index.json")

def infer_labels_from_ann_structure(ann_root: Path) -> dict[str, int]:
    """Infer {video_id: label} from annotation folder structure.

    Supported convention:
      ann_root/PASS/<video_id>/00001234.json and ann_root/FAIL/<video_id>/00001234.json

    Returns:
        Dict mapping video_id -> 1 for PASS, 0 for FAIL
    """
    labels: dict[str, int] = {}

    pass_dir = ann_root / "PASS"
    fail_dir = ann_root / "FAIL"

    if pass_dir.exists() and pass_dir.is_dir():
        for vid_dir in sorted([p for p in pass_dir.iterdir() if p.is_dir()]):
            labels[vid_dir.name] = 1

    if fail_dir.exists() and fail_dir.is_dir():
        for vid_dir in sorted([p for p in fail_dir.iterdir() if p.is_dir()]):
            labels[vid_dir.name] = 0

    return labels

def numerical_sort_key(name: str) -> int: #Splits a string into a list of strings and integers for natural sorting.
    #example: "00002274.jpg" -> 2274
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return 10**12  #Large number for non-numeric names
    
def main():
    labels = infer_labels_from_ann_structure(ann_dir)
    if not labels:
        raise FileNotFoundError(
            f"Could not infer labels. Expected folders:\n"
            f"  {ann_dir}/PASS/<video_id>/... and {ann_dir}/FAIL/<video_id>/...\n"
            f"Found neither PASS nor FAIL video folders."
        )

    items = []
    for video_id, y in labels.items():
        # Annotations live under rectangle_label_videos/PASS|FAIL/<video_id>
        cls = "PASS" if y == 1 else "FAIL"
        video_ann_dir = ann_dir / cls / video_id

        # Frames can be either raw_videos/<video_id>/... OR raw_videos/PASS|FAIL/<video_id>/...
        frames_dir = raw_dir / video_id
        candidate = raw_dir / cls / video_id
        if candidate.exists():
            frames_dir = candidate

        #Fallback Checks for directories:
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not video_ann_dir.exists():
            raise FileNotFoundError(f"Annotation frames directory not found: {video_ann_dir}")
        
        frames = sorted([p.name for p in frames_dir.glob("*.jpg")], key=numerical_sort_key)
        if len(frames) == 0:
            raise ValueError(f"No frames found in directory: {frames_dir}")
        
        #Sanity check: Ensure each frame has a corresponding annotation file
        annotated_frames = []
        for fn in frames:
            jf = video_ann_dir / (Path(fn).stem + ".json")
            if not jf.exists():
                annotated_frames.append(jf)
            
        if len(annotated_frames) == 0:
            raise RuntimeError(f"No annotated frames found for video")
        
        items.append({
            "video_id": video_id,
            "frames": frames,
            "label": y,
            "frames_dir": str(frames_dir),
            "ann_dir": str(video_ann_dir),
            "annotated_frames": annotated_frames
        })
#Ensure output directory exists!
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(items, indent=2))#Write the collected data to a JSON file.
    print(f"Index written to {out_json} with {len(items)} videos.")

#Entry point / main function call     
if __name__ == "__main__":
    main()
