import os, json, csv, argparse, sys
from typing import Dict, List, Optional
import numpy as np
# Ensure `<repo_root>/src` is on sys.path so we can import run_SVM config reliably
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from run_SVM import config


#Biomarkers of interest and their label synonyms in case we misspelled them during annotation.
BIOMARKERS = ["vocal_cords", "epiglottis", "arytenoids", "esophagus", "endotracheal_tube"]

_LABEL_SYNONYMS = {
    #Vocal cords
    "vocalcords": "vocal_cords",
    "vocal_cords": "vocal_cords",
    "vocal cords": "vocal_cords",
    "cords": "vocal_cords",
    "glottis": "vocal_cords",

    #Epiglottis
    "epiglottis": "epiglottis",

    #Arytenoids
    "arytenoid": "arytenoids",
    "arytenoids": "arytenoids",

    #Esophagus
    "esophagus": "esophagus",
    "oesophagus": "esophagus",

    #Endotracheal tube
    "ett": "endotracheal_tube",
    "endotracheal_tube": "endotracheal_tube",
    "endotracheal tube": "endotracheal_tube",
    "tube": "endotracheal_tube",
}

#Ensuring the directory for the output CSVs exists:
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

#Reading our index json file.
def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

#Writing the biomarker/annotation summary to a csv file.
def _write_csv(path: str, rows: List[Dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

#Canonicalize label strings to handle common synonyms and misspellings.
def _canon_label(label: str) -> str:
    s = str(label).strip().lower()
    return _LABEL_SYNONYMS.get(s, s)

#Extrating the labels from the annotation json files.
def _extract_labels_from_ann_json(json_path: str) -> List[str]:
    """
    Supports LabelMe-style:
      { "shapes": [ {"label": "..."} , ... ] }
    and a couple generic fallbacks.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return []
    labels: List[str] = []

    #LabelMe-style, which is what we used for annotation in our project. 
    if isinstance(data, dict) and isinstance(data.get("shapes"), list):
        for sh in data["shapes"]:
            if isinstance(sh, dict) and "label" in sh:
                labels.append(str(sh["label"]))

    #Generic fallback, in case some annotation files have a different structure:
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        for ann in data["annotations"]:
            if isinstance(ann, dict):
                if "label" in ann:
                    labels.append(str(ann["label"]))
                elif "category" in ann:
                    labels.append(str(ann["category"]))
                elif "category_name" in ann:
                    labels.append(str(ann["category_name"]))

    #Flat list fallback, in case some annotation files are just a list of labels dicts:
    if isinstance(data, list):
        for ann in data:
            if isinstance(ann, dict) and "label" in ann:
                labels.append(str(ann["label"]))

    return labels

#Computing the presence rate of each biomarker/annotation across the video frames, based ON the annotation json files!
def _video_presence_rates(ann_dir: Optional[str]) -> Dict[str, float]:
    """
    Presence rate for each biomarker:
      (# JSON frames containing biomarker) / (# JSON frames)
    """ 
    #If no annotation directory or it doesn't exist, we return nan for all biomarkers/annotations
    #Which defaults to treating them as unknown/absent in the .csv summary output.
    if not ann_dir or not os.path.isdir(ann_dir):
        return {b: float("nan") for b in BIOMARKERS}

    json_files = [fn for fn in os.listdir(ann_dir) if fn.lower().endswith(".json")]
    if not json_files:
        return {b: float("nan") for b in BIOMARKERS}
    
    counts = {b: 0 for b in BIOMARKERS}
    total = 0

    for fn in json_files:
        jp = os.path.join(ann_dir, fn)
        total += 1
        raw = _extract_labels_from_ann_json(jp)
        canon = {_canon_label(l) for l in raw}
        for b in BIOMARKERS:
            if b in canon:
                counts[b] += 1

    if total == 0:
        return {b: float("nan") for b in BIOMARKERS}

    return {b: counts[b] / float(total) for b in BIOMARKERS}

#Loading the index.json file to get the annotation directories for each video, which we need to compute the biomarker presense rates.
def _load_index_by_vid() -> Dict[str, Dict]:
    index_items = _read_json(config.index_json_path)
    out = {}
    for it in index_items:
        if isinstance(it, dict) and "video_id" in it:
            out[str(it["video_id"])] = it
    return out

#Utility to read csv file into a list of dicts, which is the format we use for the eval outputs and biomarker summaries. 
#list of dicts is easier to work with for our purpose than pandas DataFrames, and we want to avoid adding more dependencies in post-proc scripts!
def _read_csv(path: str) -> List[Dict]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

#Utilitiy to get the first non-empty value from a list of possible keys in a dict.
def _get_first_key(row: Dict, keys: List[str], default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default

def _get_video_id(row: Dict) -> Optional[str]:
    """Return a best-effort video identifier from an eval row."""
    vid = _get_first_key(row, ["video_id", "vid", "video", "id", "sample_id"], default=None)
    if vid is None:
        return None
    return str(vid)

#Main function to load eval outputs, compute biomarker/annotation presence rates, and summarize model performance stratified by biomarker/annotation presence.
def _load_eval_rows(report_dir: str, fold: Optional[int], results_csv: Optional[str] = None) -> List[Dict]:
    """Load video-level prediction rows produced by eval.py.

    Priority:
      1) Explicit --results_csv path
      2) Common aggregate filenames in report_dir
      3) fold_<k>_results.csv (single fold)
      4) If fold is None, auto-aggregate all fold_<k>_results.csv files
    """
    #1) Explicit override
    if results_csv:
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"Missing results CSV: {results_csv}")
        return _read_csv(results_csv)

    #2) Common aggregate names
    if fold is None:
        candidates = [
            "all_folds_results.csv",
            # legacy / fallbacks (not ideal for biomarker eval)
            "all_folds_metrics.csv",
            "all_folds_metrics_by_folds.csv",
            "all_folds_metrics_by_fold.csv",
        ]
        for name in candidates:
            path = os.path.join(report_dir, name)
            if os.path.exists(path):
                return _read_csv(path)

    #3) Single-fold file
    if fold is not None:
        path = os.path.join(report_dir, f"fold_{fold}_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run eval.py for fold {fold}, or pass --results_csv to biomarker_eval.py"
            )
        return _read_csv(path)

    #4) Aggregate fold_<k>_results.csv across all folds
    rows: List[Dict] = []
    found_any = False
    for k in range(int(config.kfolds)):
        p = os.path.join(report_dir, f"fold_{k}_results.csv")
        if os.path.exists(p):
            found_any = True
            these = _read_csv(p)
            for r in these:
                r.setdefault("fold", str(k))
            rows.extend(these)
    if not found_any:
        raise FileNotFoundError(
            f"No eval output CSVs found in {report_dir}. Expected one of: all_folds_metrics*.csv or fold_<k>_results.csv. Run eval.py first."
        )
    return rows

#Convert values to int or float with error hanlding, used since some of the output columns might be strings
def _to_int(x) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(x)

#Convert values to float with error handling, used since some of the output columns might be strings
def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

#Core function to summarize biomarker/annotation presense and model performance stratified by biomarker/annotation presence.
def _summarize_biomarkers(rows: List[Dict], index_by_vid: Dict[str, Dict]) -> List[Dict]:
    """
    rows must include:
      video_id, true_label, predicted_label, fold (optional)
    """
    # Compute per-video presence rates
    presence: Dict[str, Dict[str, float]] = {}
    for r in rows:
        vid = _get_video_id(r)
        if vid is None:
            # Likely we loaded a metrics CSV instead of a results CSV.
            continue
        if vid in presence:
            continue
        meta = index_by_vid.get(vid, {})
        ann_dir = meta.get("ann_dir") or meta.get("annotated_dir")
        presence[vid] = _video_presence_rates(ann_dir)

    out: List[Dict] = []
#For each biomarker/annotation, we compute # of videos with annotation present, acc when present, # of vid with anotation absent, acc when absent...
#mean presence rate across vids, and Conf Matrix stats when present.

    for b in BIOMARKERS:
        present_total = 0
        present_correct = 0
        absent_total = 0
        absent_correct = 0

        tp = tn = fp = fn = 0
        prs = []

        for r in rows:
            vid = _get_video_id(r)
            if vid is None:
                continue
            yt_raw = _get_first_key(r, ["true_label", "y_true", "label", "target"], default=None)
            yp_raw = _get_first_key(r, ["predicted_label", "y_pred", "pred", "prediction"], default=None)
            if yt_raw is None or yp_raw is None:
                #If eval rows don’t have the expected columns, skip this row
                continue
            yt = _to_int(yt_raw)
            yp = _to_int(yp_raw)
            correct = 1 if yt == yp else 0

            pr = presence.get(vid, {}).get(b, float("nan"))
            if not np.isnan(pr):
                prs.append(float(pr))

            is_present = (not np.isnan(pr)) and (pr > 0.0)

            if is_present:
                present_total += 1
                present_correct += correct

                if yt == 1 and yp == 1:
                    tp += 1
                elif yt == 0 and yp == 0:
                    tn += 1
                elif yt == 0 and yp == 1:
                    fp += 1
                elif yt == 1 and yp == 0:
                    fn += 1
            else:
                absent_total += 1
                absent_correct += correct

        out.append(
            {
                "biomarker": b,
                "n_videos_present": present_total,
                "acc_when_present": (present_correct / present_total) if present_total > 0 else float("nan"),
                "n_videos_absent_or_unknown": absent_total,
                "acc_when_absent_or_unknown": (absent_correct / absent_total) if absent_total > 0 else float("nan"),
                "mean_presence_rate_over_videos": float(np.nanmean(np.array(prs, dtype=float))) if prs else float("nan"),
                "tp_when_present": tp,
                "tn_when_present": tn,
                "fp_when_present": fp,
                "fn_when_present": fn,
            }
        )

    return out


def run_biomarker_eval(
    fold: Optional[int] = None,
    report_dir: Optional[str] = None,
    results_csv: Optional[str] = None,
) -> List[str]:
    """Run biomarker-stratified evaluation and return written CSV paths."""
    rd = report_dir or config.reports_dir

    print(f"[biomarker_eval] using report_dir={rd}")
    if results_csv:
        print(f"[biomarker_eval] using results_csv={results_csv}")

    index_by_vid = _load_index_by_vid()

    if fold is None:
        rows = _load_eval_rows(rd, fold=None, results_csv=results_csv)
        summary = _summarize_biomarkers(rows, index_by_vid)
        if not summary:
            raise RuntimeError(
                "No biomarker summary rows were produced. This usually means the input CSV does not contain per-video predictions. "
                "Point --results_csv to a fold_<k>_results.csv or all_folds_results.csv produced by eval.py."
            )
        out_csv = os.path.join(rd, "biomarkers_summary_all.csv")
        _write_csv(out_csv, summary)
        print(f"[biomarker_eval] wrote {out_csv}")
        return [out_csv]

    rows = _load_eval_rows(rd, fold=fold, results_csv=results_csv)
    summary = _summarize_biomarkers(rows, index_by_vid)
    if not summary:
        raise RuntimeError(
            "No biomarker summary rows were produced. This usually means the input CSV does not contain per-video predictions. "
            "Point --results_csv to a fold_<k>_results.csv or all_folds_results.csv produced by eval.py."
        )
    out_csv = os.path.join(rd, f"biomarkers_summary_fold_{fold}.csv")
    _write_csv(out_csv, summary)
    print(f"[biomarker_eval] wrote {out_csv}")
    return [out_csv]


def main() -> None:
    ap = argparse.ArgumentParser(description="Biomarker-stratified post-processing for 3DCNN eval outputs")
    ap.add_argument("--fold", type=int, default=None, help="Evaluate biomarker stats for one fold only.")
    ap.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Optional: path to a specific eval CSV (overrides auto-detection).",
    )
    ap.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Optional: override report directory (default: <runs_path>/reports).",
    )
    args = ap.parse_args()

    run_biomarker_eval(fold=args.fold, report_dir=args.report_dir, results_csv=args.results_csv)


if __name__ == "__main__":
    main()
""" Expected outputs after running this script:
- CSV summaries per fold + overall, e.g.:
1) Presence Rate: Videos with biomarker present / total videos
2) Accuracy when present: Correct predictions / total videos with biomarker present
3) Accuracy when absent/unknown: Correct predictions / total videos without biomarker or unknown presence
4) Accuracy difference: Accuracy when present - Accuracy when absent/unknown
5) Misclassification rate when present: Incorrect predictions / total videos with biomarker present"""
