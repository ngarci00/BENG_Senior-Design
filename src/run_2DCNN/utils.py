import os, random, torch

def set_seed(seed):#setting the seed for reproducibility 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid(x):#sigmoid function, for converting logits to probabilities
    return 1 / (1 + torch.exp(-x))

@torch.no_grad()  # decorator to disable gradient calculation, for inference
def compute_binary_metrics(logits, targets):
    """Compute binary classification metrics.

    Supports:
      - logits shape [B] or [B,1]: treated as single-logit binary classifier (sigmoid)
      - logits shape [B,2]: treated as 2-class logits (softmax)

    Returns dict with accuracy, balanced_accuracy, precision, recall, f1_score.
    """
    #Ensuring the inputs are tensor
    if not torch.is_tensor(logits): 
        logits = torch.tensor(logits)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    #Targets must be 0/1 integers
    y = targets.long().view(-1)

    #Normalize logits into probabilities for the positive class
    if logits.ndim == 2 and logits.size(1) == 2:
        #Two-class logits
        probs_pos = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1).long()
    else:
        #Single-logit binary classifier: allow [B,1] or [B]
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        logits = logits.view(-1)
        probs_pos = sigmoid(logits)
        preds = (probs_pos >= 0.5).long()

    #Confusion matrix components
    true_positives = int(((preds == 1) & (y == 1)).sum().item())
    true_negatives = int(((preds == 0) & (y == 0)).sum().item())
    false_positives = int(((preds == 1) & (y == 0)).sum().item())
    false_negatives = int(((preds == 0) & (y == 1)).sum().item())

    denom = max(true_positives + true_negatives + false_positives + false_negatives, 1) 
    accuracy = (true_positives + true_negatives) / denom #Overall accuracy: (TP + TN) / Total

    recall = true_positives / max(true_positives + false_negatives, 1)  # TPR
    specificity = true_negatives / max(true_negatives + false_positives, 1)  # TNR
    balanced_accuracy = (recall + specificity) / 2 #Balanced accuracy: average of TPR and TNR, useful for imbalanced datasets <-

    precision = true_positives / max(true_positives + false_positives, 1)
    f1_score = (2 * precision * recall) / max(precision + recall, 1e-8)

    #Return all metrics in a dictionary, including confusion matrix components for optional deeper analysis
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        # Optional extras (kept lightweight)
        "tp": true_positives,
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
    }

def ensure_dir_exists(dir_path): #function to ensure that a directory exists
    os.makedirs(dir_path, exist_ok=True)
