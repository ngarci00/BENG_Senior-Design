import os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from config import (runs_path, epochs, batch_size, learning_rate, weight_decay, num_workers, 
                    stop_if_val_acc_perfect, perfect_acc_tolerance, seed)
from dataset import VideoFrameDataset
from model import build_model
from utils import set_seed, compute_binary_metrics, ensure_dir_exists

def train_2dcnn(fold, device):
    set_seed(seed + fold)#set the seed for reproducibility, adding fold to ensure different seeds for different folds

    output_dir = os.path.join(runs_path, f"fold_{fold}")#define the output directory for the current fold
    ensure_dir_exists(output_dir)#ensure the output directory exists

    #Creating the datasets (ds) and dataloaders (dl) for training and validation:
    ds_train = VideoFrameDataset(fold=fold, split='train', seed=seed + fold)#create the training dataset
    ds_val = VideoFrameDataset(fold=fold, split='val', seed=seed + fold)#create the validation dataset

    #Create the validation dataset
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True )#create the training dataloader 
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=True)#create the validation dataloader
    #num_workers refers to the number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    #if we increase num_workers, it can speed up data loading by using multiple subprocesses, but it may also increase memory usage.

    model = build_model().to(device) #build the 3D CNN model and move it to the specified device (CPU or GPU)
    criterion_ce = nn.CrossEntropyLoss() #define the loss function (Cross Entropy Loss)
    criterion_bce = nn.BCEWithLogitsLoss() #define the loss function (Binary Cross Entropy with Logits Loss)
    #BCEWithLogitsLoss is used for binary classification tasks, while CrossEntropyLoss is used for multi-class classification tasks.
    #Our application only needs binary classification but we have both as a fallback in case we want to experiment with multi-class classification in the future.

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #define the optimizer (AdamW) with the model parameters, learning rate, and weight decay

    def _is_two_class_logits(t: "torch.Tensor") -> bool:
        return t.ndim == 2 and t.size(1) == 2
    
    def _compute_loss(logits: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        if _is_two_class_logits(logits):
            return criterion_ce(logits, y.long())#if the logits are for two classes, use cross entropy loss
        return criterion_bce(logits.squeeze(-1), y.float())#if the logits are for binary classification, use binary cross entropy loss
        
    def _positive_logit(logits: "torch.Tensor") -> "torch.Tensor":
        if _is_two_class_logits(logits):
            return logits[:, 1]#if the logits are for two classes, return the logits for the positive class
        return logits.squeeze(-1)#if the logits are for binary classification, return the logits as is


    best_f1_score = -1#initialize the best F1 score to -1 being the lowest possible value
    best_val_accuracy = 0#initialize the best validation accuracy to 0
    best_path = os.path.join(output_dir, "best_model.pt")#define the path to save the best model

    #Here is the bread and butter; aka the training loop (:
    for epoch in range(1, epochs + 1):
        model.train()#Setting the model to training mode
        running = 0#initializing the running loss to 0

        #TRAINING PHASE
        for x, y, _ in dl_train: #iterating over dl_train, which gives us batches of data x, labels y, and video names (ignored here)
            x = x.to(device) #move the input data to the specified device
            if x.ndim != 5:
                raise RuntimeError(f"Exppected batched frames with shape (B, C, T, H, W), but got {tuple(x.shape)}")#check if the input data has the expected shape (B, C, T, H, W)
            y = y.to(device) #move the labels to the specified device

            logits = model(x) #forward pass: passes the input data through the model to get the output logits
            loss = _compute_loss(logits, y) #compute the loss by comparing the logits with true labels y

            optimizer.zero_grad()#zero the gradients before backpropagation
            loss.backward()#backpropagation: compute the gradients of the loss with respect to the model parameters
            optimizer.step()#update the model parameters based on the computed gradients

            running +=  float(loss.item()) #accumulate the loss for the current batch into the running loss
        training_loss = running / max(len(dl_train), 1) #calculate the average training loss for the epoch

        #VALIDATION PHASE
        model.eval()#Setting the model to evaluation mode
        all_logits = []#initialize a list to store all the logits for the validation set
        all_targets = []#initialize a list to store all the targets for the validation set

        with torch.no_grad():#disable gradient calculation for validation
            for x, y, _ in dl_val:
                x = x.to(device)
                if x.ndim != 5:
                    raise RuntimeError(f"Exppected batched frames with shape (B, C, T, H, W), but got {tuple(x.shape)}")#check if the input data has the expected shape (B, C, T, H, W)
        
                y = y.to(device)
                logits = model(x)
                all_logits.append(logits.detach().cpu())#append the logits to the list moving to CPU
                all_targets.append(y.detach().cpu())#append the targets to the list, moving to CPU

        logits = torch.cat(all_logits, dim=0)#concatenate all the logits along the batch dimension
        #concatenating means that we are combining all the logits from different batches into a single tensor
        targets = torch.cat(all_targets, dim=0)#concatenate all the targets along the batch dimension

        metrics = compute_binary_metrics(_positive_logit(logits), targets) #compute the binary classification metrics using the logits and targets
        val_accuracy = metrics["accuracy"] #extract the validation accuracy from the computed metrics

        #Some summary prints for the current epoch:
        print(f"[Fold {fold}, Epoch {epoch:02d}/{epochs}]")
        print(f"Training Loss: {training_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}, F1 Score: {metrics['f1_score']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")

        #Here we check if the current model is the best one based on F1 score... Which can save us from overfitting by accuracy alone:
        if metrics["f1_score"] > best_f1_score:
            best_f1_score = metrics["f1_score"]#update the best F1 score
            best_val_accuracy = max(best_val_accuracy,val_accuracy)#update the best validation accuracy
            torch.save(model.state_dict(), best_path)#save the model state dict to the best path
        
        best_val_accuracy = max(best_val_accuracy,val_accuracy)#update the best validation accuracy

        #Early STOP if we reach perfect validation accuracy for a certain number of epochs, we can stop training to prevent overfitting:
        if stop_if_val_acc_perfect and val_accuracy >= 1 - perfect_acc_tolerance:
            print(f"Fold {fold} has reached perfect validation accuracy at epoch {epoch}. Stopping training to prevent overfitting!")
            break

    #Fold summaries:
    summary = {"fold": fold, "best_f1_score": best_f1_score, "best_val_accuracy": best_val_accuracy, "best_model_path": best_path}
    #Save the summary for the current fold to a json file in the utput directory:
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary 