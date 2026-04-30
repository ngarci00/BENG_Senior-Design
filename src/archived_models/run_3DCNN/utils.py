import os, random, torch

def set_seed(seed):#setting the seed for reproducibility 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid(x):#sigmoid function, for converting logits to probabilities
    return 1 / (1 + torch.exp(-x))

@torch.no_grad()#decorator to disable gradient calculation, for inference
def compute_binary_metrics(logits,targets):

    if logits.ndim == 2:
        logits = logits.squeeze(1) #remove the extra dimension if present
    probs = sigmoid(logits) #convert logits to probabilities
    preds = (probs >= 0.5).float() #threshold probabilities to get binary predictions
    y = targets.long() #convert targets to long type for metric calculation

    true_positives = int(((preds == 1) & (y == 1)).sum().item()) #count true positives
    true_negatives = int(((preds == 0) & (y == 0)).sum().item()) #count true negatives
    false_positives = int(((preds == 1) & (y == 0)).sum().item()) #count false positives
    false_negatives = int(((preds == 0) & (y == 1)).sum().item()) #count false negatives

    accuracy = (true_positives + true_negatives) / max(true_positives + true_negatives + false_positives + false_negatives, 1) #calculate accuracy

    true_positive_rate = true_positives / max(true_positives + false_negatives, 1) #calculate true positive rate (RECALL)
    true_negative_rate = true_negatives / max(true_negatives + false_positives, 1) #calculate true negative rate (SPECIFICITY)
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2 #calculate balanced accuracy, whic is the avg of TP and TN rates

    preccision = true_positives / max(true_positives + false_positives, 1) #calculate (PRECISION)
    f1_score = (2*preccision*true_positive_rate) / max(preccision + true_positive_rate, 1e-8) #calculate the F1 SCORE

    return {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "precision": preccision, "recall": true_positive_rate, "f1_score": f1_score}

def ensure_dir_exists(dir_path):#function to ensure that a directory exists
    os.makedirs(dir_path, exist_ok=True)
