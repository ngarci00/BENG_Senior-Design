import os, json, joblib, numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
import config

#Evaluation script to evaluate the trained SVM model on the validation set for each fold
def eval_fold(fold: int) -> dict:
    output_path = os.path.join(config.features_dir, f"fold_{fold}.npz")
    data = np.load(output_path, allow_pickle=True)
    Xva, yva = data["X_val"], data["y_val"]

    model_path = os.path.join(config.models_dir, f"svm_fold_{fold}.joblib")
    model = joblib.load(model_path)

    yhat = model.predict(Xva)

    acc = accuracy_score(yva, yhat)
    balacc = balanced_accuracy_score(yva, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(yva, yhat, average="binary", zero_division=0)
    cm = confusion_matrix(yva, yhat).tolist()
    
    #Return a dictionary with all the metrics and the confusion matrix for this fold
    return {
        "fold": fold,
        "acc": float(acc),
        "bal_acc": float(balacc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

def main():
    os.makedirs(config.reports_dir, exist_ok=True)
    results = [eval_fold(f) for f in range(config.kfolds)]

    out_path = os.path.join(config.reports_dir, "svm_kfold_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SVM] Wrote: {out_path}")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()