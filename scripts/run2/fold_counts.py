#Short script to visualize the distribution of PASS/FAIL in each fold
import json
from collections import Counter

index = json.load(open("data/videos/index.json"))
splits = json.load(open("data/videos/splits.json"))

y = {x["video_id"]: int(x["label"]) for x in index}  # 1=PASS, 0=FAIL

for fold, d in splits.items():
    tr = Counter(y[v] for v in d["train"])
    va = Counter(y[v] for v in d["val"])
    print(f"{fold}  train PASS/FAIL = {tr[1]}/{tr[0]}   val PASS/FAIL = {va[1]}/{va[0]}")