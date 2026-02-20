#This Script serves as a post-processing analysis tool for any used model (3DCNN , 2DCNN, SVM, etc)
"""
Please make sure you run run.py and save the model before running this script, otherwise it will not work! 
Alsooo run in terminal with:
python src/run_3DCNN/eval.py (It should automatically read th best model from each fold and produce statistics and plots in runs/reports/)
"""
import os, torch, json, csv, argparse, numpy as np, matplotlib.pyplot as plt 
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve)
from config import splits_json_path, runs_path, kfolds, clip_len, seed
from dataset import VideoClipDataset
from typing import Dict, List, Tuple #For type hinting, helps with readability and debugging
from model import build_model

#Check directory exists, if not create it
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#Read json file
def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _write_csv(path: str, rows: List[Dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows: 
        return
    with open(path, "w", newline="") as f: #newline prevents extra blank lines 
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

#Confusion Matrix Plotting
def _save_confusion_matrix(cm: np.ndarray, out_png: str, title: str) -> None:
    _ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1],["FAIL(0)", "PASS(1)"])
    plt.yticks([0, 1],["FAIL(0)", "PASS(1)"])

    for (i,j), v in np.ndenumerate(cm):
        plt.text(j,i, str(int(v)), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

#ROC Curve Plotting
def _save_roc_curve(y_true: List[int], y_prob: List[float], out_dir:str, fold: int) -> None:
    _ensure_dir(out_dir)
    if len(set(y_true)) != 2:
        return #ROC curve only makes sense for binary classification
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr,tpr)
    plt.plot([0,1], [0,1], "--") #Diagonal line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.legend([f"ROC Curve (AUC={roc_auc:.3f})", "Predicted"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fold_{fold}_roc.png"), dpi=300)
    plt.close()

    #Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Fold {fold}: Precision-Recall Curve (AUC={pr_auc:.3f})")
    plt.legend([f"PR Curve (AUC={pr_auc:.3f})"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fold_{fold}_pr.png"), dpi=300)
    plt.close()

@torch.no_grad() #No need to compute gradients during evaluation
def _infer_fold(fold: int, device: str, eval_clips_per_video: int, batch_size: int, num_workers: int) -> Tuple[List[Dict], Dict[str, float]]:
    fold_dir = os.path.join(runs_path, f"fold_{fold}")
    best_model_path = os.path.join(fold_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found for fold {fold} at {best_model_path}")
    
    #Dataset Validation
    ds_val = VideoClipDataset(fold=fold, split="val", clip_len=clip_len, clips_per_video=eval_clips_per_video, seed=seed)

    #DataLoader Validation
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    #Model Loading
    model = build_model().to(device)
    best_model = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_model if not (isinstance(best_model, dict) and "model_state_dict" in best_model) else best_model["model_state_dict"], strict=True)
    model.eval()

    by_video_logits: Dict[str, List[float]] = {}
    by_video_labels: Dict[str, int] = {}

    #Inference Loop
    for x, y, vids in dl_val:
        x = x.to(device)
        y_int = y.detach().cpu().numpy().astype(int).reshape(-1).tolist() #Convert to numpy array of ints

        logits = model(x)
        
        if logits.ndim == 2 and logits.shape[1] == 1: #Binary classification with single logit output
            logits = logits[:, 0] #Use logit for class 0 (FAIL)
        elif logits.ndim == 2 and logits.shape[1] == 2: #Binary classification with two logit outputs
            logits = logits[:, 1] #Use logit for class 1 (PASS)
        elif logits.ndim == 1: #Already in shape (batch_size)
            pass
        else:
            raise ValueError(f"Unexpected logit shape: {tuple(logits.shape)}") #Assume it's already the correct shape
        
        logits_list = logits.detach().cpu().numpy().reshape(-1).tolist() #Convert to list of floats
        vids_list = list(vids) #Convert to list of strings

        for vid, yi, li, in zip(vids_list, y_int, logits_list):
            by_video_logits.setdefault(vid, []).append(float(li)) #Append logit to list for this video
            by_video_labels[vid] = int(yi) #Set label for this video (same for all clips)

    #Aggregate results by video
    rows:  List[Dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    for vid in sorted(by_video_logits.keys()):
        mean_logit = float(np.mean(by_video_logits[vid])) #Average logit across clips for this video
        prob = float(1/(1+np.exp(-mean_logit))) #Convert logit to probability using sigmoid
        pred = 1 if prob >= 0.5 else 0 #Threshold at 0.5 for binary classification
        yt = int(by_video_labels[vid]) #True label for this video

        #Results for this video(s)
        rows.append(
        {               
            "fold": fold,
            "video_id": vid,
            "true_label": yt,
            "predicted_label": pred,
            "predicted_prob": prob,
            "logit_mean": mean_logit,
            "n_clips": len(by_video_logits[vid]),
            }
        )
        y_true.append(yt)
        y_pred.append(pred)
        y_prob.append(prob)

    #Confusion matrix:
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() 

    metrics = {
        "fold": float(fold),
        "n_videos": float(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "bal_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }

    if len(set(y_true)) == 2: #Only plot ROC curve if we have both classes present
        fpr, tpr, _ = roc_curve(y_true, y_prob) #False positive rate, true positive rate for ROC curve
        metrics["roc_auc"] = float(auc(fpr, tpr)) #Area under ROC curve
        prec , rec, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc"] = float(auc(rec, prec))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return rows, metrics

def _add_mean_std(metrics: List[Dict]) -> List[Dict]:
    if not metrics:
        return [] #No metrics to process
    
    keys = [k for k in metrics[0].keys() if k != "fold"] #Exclude fold from mean/std calculation
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}

    for k in keys: 
        values = np.array([float(m.get(k, np.nan)) for m in metrics], dtype=float)
        mean_row[k] = float(np.nanmean(values)) #Mean of this metric across folds
        std_row[k] = float(np.nanstd(values)) #Standard deviation of this metric across

    return metrics + [mean_row, std_row] #Append mean and std rows to the list of metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Evaluate only one fold")
    parser.add_argument("--device", type=str, default=None, help="CPU or CUDA device to use for evaluation")
    parser.add_argument("--eval_clips_per_video", type=int, default=5, help="Number of clips to sample per video during evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading during evaluation")
    args = parser.parse_args()

    if args.device is None:
        device = "mps" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device.lower()
        if device not in ["cpu", "mps", "cuda"]:
            raise ValueError("Invalid device specified. Use 'CPU' or 'CUDA'.")
            

    _read_json(splits_json_path) #Ensure splits json is loaded before evaluation

    folds = [args.fold] if args.fold is not None else list(range(int(kfolds))) #Evaluate specified fold or all folds

    report_dir =os.path.join(runs_path, "reports") # <---Directory to save evaluation reports and plots, we can change the name here
    _ensure_dir(report_dir)
    
    all_rows: List[Dict] = []
    all_metrics: List[Dict] = []

    for fold in folds:
        print(f"Evaluating fold_{fold} on device: {device} ... ")
        rows, metrics = _infer_fold(fold=fold, device=device, eval_clips_per_video=args.eval_clips_per_video, batch_size=args.batch_size, num_workers=args.num_workers)

        _write_csv(os.path.join(report_dir, f"fold_{fold}_results.csv"), rows) #Save results for this fold

        y_true = [int(r["true_label"]) for r in rows]
        y_pred = [int(r["predicted_label"]) for r in rows]
        y_prob = [float(r["predicted_prob"]) for r in rows]

        #Save confusion matrix plot for this fold
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        _save_confusion_matrix(cm, os.path.join(report_dir, f"fold_{fold}_confusion_matrix.png"), title=f"Fold {fold} Confusion Matrix")
        #Save ROC curve plot for this fold
        _save_roc_curve(y_true, y_prob, report_dir, fold)
        all_rows.extend(rows) #Add this fold's results to the overall list
        all_metrics.append(metrics) #Add this fold's metrics to the overall list

        print(
            f"Fold {fold} Metrics: Accuracy={metrics['accuracy']:.3f}, Balanced Accuracy={metrics['bal_accuracy']:.3f}, F1 Score={metrics['f1_score']:.3f}" 
            f"True Positives={metrics['tp']}, True Negatives={metrics['tn']}, False Positives={metrics['fp']}, False Negatives={metrics['fn']}")
        
            # Write combined results and summary after all folds are processed
        _write_csv(os.path.join(report_dir, "all_folds_metrics.csv"), all_rows) #Save combined results for all folds
        _write_csv(os.path.join(report_dir, "all_folds_metrics_by_folds.csv"), all_metrics) #Save combined metrics for all folds
        _write_csv(os.path.join(report_dir, "all_folds_summary.csv"), _add_mean_std(all_metrics)) #Save summary metrics with mean and std across folds

        with open(os.path.join(report_dir, "all_folds_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Evaluation complete. Results saved to {report_dir}    ⸜(｡˃ ᵕ ˂ )⸝♡ ")
        
if __name__ == "__main__":
    main()

"""These are the expected outputs after running this script:
- Per-fld Confusion Matrix Plots
- ROC Curves
- Summary Metrics:

1) Accuracy (acc) : (TP + TN) / Total
2) Balanced Accuracy (bal_acc) : (TPR + TNR) / 2 helps 
3) F1 Score (f1) : 2 * (Precision * Recall) / (Precision + Recall)
4) Precision (prec) : TP / (TP + FP)
5) Recall (rec) : TP / (TP + FN)
6) ROC Curve + AUC (roc_auc) : Area under the ROC curve, which plots TPR vs FPR at various thresholds"""
