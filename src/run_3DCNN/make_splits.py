import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

INDEX_JSON = Path("data/videos/index.json")
OUT_SPLITS = Path("data/videos/splits.json")

def main(k=5, seed=42): #k-fold of 5, seed is used for reproducibility, meaning the same splits will be generated each time
    items = json.loads(INDEX_JSON.read_text())
    video_ids = [x["video_id"] for x in items]
    labels = [x["label"] for x in items]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(video_ids, labels)):
        splits[f"fold_{fold}"] = {"train": [video_ids[i] for i in train_idx], "val": [video_ids[i] for i in val_idx]}

        OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
        OUT_SPLITS.write_text(json.dumps(splits, indent=2))
        print(f"Splits saved to {OUT_SPLITS}, with {k} folds.")

if __name__ == "__main__":
    main()