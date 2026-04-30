import os, json, argparse, random
from sklearn.model_selection import StratifiedKFold
from config import index_json_path, splits_json_path, kfolds, seed

def main():

    parser = argparse.ArgumentParser(description="Create stratified k-fold splits for 3D-CNN videos!")
    parser.add_argument("--index", default=index_json_path, help="Path to index JSON (list of {video_id,label,...}).")
    parser.add_argument("--out", default=splits_json_path, help="Output path for splits JSON.")
    parser.add_argument("--kfolds", type=int, default=int(kfolds), help="Number of folds (K) for StratifiedKFold.")
    parser.add_argument("--seed", type=int, default=int(seed), help="Random seed.")
    parser.add_argument("--subset-per-class", type=int, default=None,
                        help="If set, build splits using only this many videos per class (balanced). Example: 5 -> 5 PASS and 5 FAIL.")
    args = parser.parse_args()

    with open(args.index, "r") as f:
        items = json.load(f)

    #Create a list of video IDs and labels for stratified splitting    
    video_ids = [x["video_id"] for x in items]
    labels = [x["label"] for x in items]

    n_pos = sum(y == 1 for y in labels)
    n_neg = sum(y == 0 for y in labels)

    # Print dataset stats (do not hard-fail on imbalance; we may subset below)
    print(f"Total videos in index: {len(video_ids)}, PASS: {n_pos}, FAIL: {n_neg}")

    # Optional: build a balanced subset for quick experiments (e.g., 5 PASS + 5 FAIL)
    if args.subset_per_class is not None:
        subset_n = int(args.subset_per_class)
        if subset_n <= 0:
            raise ValueError("--subset-per-class must be a positive integer")

        pos_ids = [vid for vid, y in zip(video_ids, labels) if y == 1]
        neg_ids = [vid for vid, y in zip(video_ids, labels) if y == 0]

        if len(pos_ids) < subset_n or len(neg_ids) < subset_n:
            raise ValueError(
                f"Not enough samples for requested subset: need {subset_n} per class, "
                f"but have PASS={len(pos_ids)} and FAIL={len(neg_ids)}"
            )

        rng = random.Random(args.seed)
        rng.shuffle(pos_ids)
        rng.shuffle(neg_ids)

        video_ids = pos_ids[:subset_n] + neg_ids[:subset_n]
        labels = [1] * subset_n + [0] * subset_n
        rng.shuffle(video_ids)

        # Rebuild labels to match shuffled video_ids
        label_map = {vid: 1 for vid in pos_ids[:subset_n]}
        label_map.update({vid: 0 for vid in neg_ids[:subset_n]})
        labels = [label_map[vid] for vid in video_ids]

        print(f"Using balanced subset: {len(video_ids)} videos (PASS={subset_n}, FAIL={subset_n})")

    k = int(args.kfolds)

    n_pos_cur = sum(y == 1 for y in labels)
    n_neg_cur = sum(y == 0 for y in labels)
    min_class = min(n_pos_cur, n_neg_cur)
    if k > min_class:
        raise ValueError(f"Error: Number of folds (k={k}) cannot be greater than number of samples in the smallest class (n={min_class}).")
    if k < 2:
        raise ValueError(f"Error: Number of folds (k={k}) must be at least 2.")
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(args.seed))

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(video_ids, labels)):
        splits[f"fold_{fold}"] = {
            "train": [video_ids[i] for i in train_idx],
            "val": [video_ids[i] for i in val_idx],
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {k}-fold splits to {args.out}")

if __name__ == "__main__":
    main()