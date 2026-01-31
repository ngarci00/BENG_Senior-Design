#The goal of this script is to ensure all frames have matching JSON files, before any training is done.
import json
from pathlib import Path
import csv

#Directories and file paths
raw_dir = Path("data/videos/raw_videos")
ann_dir = Path("data/videos/rectangle_label_videos")
labels_csv = Path("data/videos/video_labels.csv")
out_json = Path("data/videos/index.json")

def get_video_labels(labels_csv: Path) -> dict:
    """Read CSV with columns: video_id,label and return {video_id: int(label)}."""
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {labels_csv}. Create it at data/videos/video_labels.csv "
            f"with columns: video_id,label"
        )

    labels: dict[str, int] = {}
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Labels CSV must have columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            vid = row["video_id"].strip()
            if not vid:
                continue
            labels[vid] = int(row["label"])
    return labels

def numerical_sort_key(name: str) -> int: #Splits a string into a list of strings and integers for natural sorting.
    #example: "00002274.jpg" -> 2274
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return 10**12  #Large number for non-numeric names
    
def main():
    labels = get_video_labels(labels_csv)

    items = []
    for video_id, y in labels.items():
        frames_dir = raw_dir / video_id
        video_ann_dir = ann_dir / video_id
        #Fallback Checks for directories:
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not video_ann_dir.exists():
            raise FileNotFoundError(f"Annotation frames directory not found: {video_ann_dir}")
        
        frames = sorted([p.name for p in frames_dir.glob("*.jpg")], key=numerical_sort_key)
        if len(frames) == 0:
            raise ValueError(f"No frames found in directory: {frames_dir}")
        
        #Sanity check: Ensure each frame has a corresponding annotation file
        missing_annotations = []
        for fn in frames:
            jf = video_ann_dir / (Path(fn).stem + ".json")
            if not jf.exists():
                missing_annotations.append(jf.name)
            
        if missing_annotations:
            raise RuntimeError(f"Missing annotation files for video {video_id}: {missing_annotations}")
        
        items.append({
            "video_id": video_id,
            "frames": frames,
            "label": y,
            "frames_dir": str(frames_dir),
            "ann_dir": str(video_ann_dir)
        })
#Ensure output directory exists!
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(items, indent=2))#Write the collected data to a JSON file.
    print(f"Index written to {out_json} with {len(items)} videos.")

#Entry point / main function call     
if __name__ == "__main__":
    main()


