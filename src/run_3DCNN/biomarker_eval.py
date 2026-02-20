import os, json, csv, argparse, numpy as np 
from typing import List, Dict, Optional
from config import runs_path, kfolds, index_json_path

BIOMARKER_NAMES = ["vocal_cords", "epiglottis", "esophagus", "arytenoids", "endotracheal_tube"]

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _write_csv(path: str, rows: List[Dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def _extract_labels_from_annotations_json(json_path: str) -> List[str]:
    try: 