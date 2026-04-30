#Short script to visualize the distribution of PASS/FAIL in each fold
import json
from collections import Counter #to count the number of PASS/FAIL in each fold
from config import index_json_path, splits_json_path

def main():
    index = json.load(open(index_json_path))
    splits = json.load(open(splits_json_path))
    y = {x["video_id"]: int(x["label"]) for x in index}

    for fold, d in splits.items():
        train = Counter(y[v] for v in d["train"])#Count the number of PASS/FAIL in the training set
        val = Counter(y[v] for v in d["val"])#Count the number of PASS/FAIL in the validation set
        print(f"Fold {fold} Training set: PASS/FAIL = {train[1]}/{train[0]}  Validation set: PASS/FAIL = {val[1]}/{val[0]}")

if __name__ == "__main__":
    main()