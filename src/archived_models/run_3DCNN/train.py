import os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from config import (runs_path, epochs, batch_size, grad_accum_steps, learning_rate, weight_decay, 
                    clip_len, clips_per_video_train, clips_per_video_val,
                    stop_if_val_acc_perfect, perfect_acc_tolerance, seed, use_mps_mixed_precision)
from dataset import VideoClipDataset
from model import build_model
from utils import set_seed, compute_binary_metrics, ensure_dir_exists


def _is_two_class_logits(t: torch.Tensor) -> bool:
    return t.ndim == 2 and t.size(1) == 2


def train_3dcnn(fold, device):
    set_seed(seed + fold)  # set the seed for reproducibility, adding fold to ensure different seeds for different folds

    output_dir = os.path.join(runs_path, f"fold_{fold}")  # define the output directory for the current fold
    ensure_dir_exists(output_dir)  # ensure the output directory exists

    effective_batch_size = 1 if device == "mps" else batch_size
    grad_steps = max(1, grad_accum_steps)
    effective_batch_size = max(1, effective_batch_size)
    if device == "mps":
        if batch_size != effective_batch_size:
            print(f"[Fold {fold}] MPS detected: using batch_size={effective_batch_size} (config was {batch_size}).")
        print(f"[Fold {fold}] MPS detected: using grad_accum_steps={grad_steps}.")

    ds_train = VideoClipDataset(fold=fold, split="train", clip_len=clip_len, clips_per_video=clips_per_video_train)  # create the training dataset
    ds_val = VideoClipDataset(fold=fold, split="val", clip_len=clip_len, clips_per_video=clips_per_video_val)  # create the validation dataset

    dl_train = DataLoader(
        ds_train,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )  # create the training dataloader
    dl_val = DataLoader(
        ds_val,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )  # create the validation dataloader

    model = build_model().to(device)  # build the 3D CNN model and move it to the specified device (CPU or GPU)
    criterion_bce = nn.BCEWithLogitsLoss()  # define the loss function (Binary Cross Entropy with Logits)
    criterion_ce = nn.CrossEntropyLoss()  # define the loss function (Cross Entropy Loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # define the optimizer

    def _compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if _is_two_class_logits(logits):
            return criterion_ce(logits, y.long())  # if the logits are for two classes, use cross entropy loss
        return criterion_bce(logits.squeeze(-1), y.float())  # if the logits are for binary classification, use binary cross entropy loss

    def _positive_logit(logits: torch.Tensor) -> torch.Tensor:
        if _is_two_class_logits(logits):
            return logits[:, 1]  # if the logits are for two classes, return the logits for the positive class
        return logits.squeeze(-1)  # if the logits are for binary classification, return the logits as is

    best_f1_score = 0  # initialize the best F1 score to -1 being the lowest possible value
    best_val_accuracy = 0  # initialize the best validation accuracy to 0
    best_path = os.path.join(output_dir, "best_model.pt")  # define the path to save the best model

    use_amp = device == "mps" and bool(use_mps_mixed_precision)

    # Here is the bread and butter; aka the training loop
    for epoch in range(1, epochs + 1):
        model.train()  # setting the model to training mode
        running = 0  # initializing the running loss to 0
        optimizer.zero_grad(set_to_none=True)

        # TRAINING PHASE
        for step, (x, y, _) in enumerate(dl_train):
            x = x.to(device, non_blocking=True)  # move the input data to the specified device
            y = y.to(device, non_blocking=True)  # move the labels to the specified device

            with torch.amp.autocast(device_type="mps", dtype=torch.float16, enabled=use_amp):
                logits = model(x)  # forward pass
                loss = _compute_loss(logits, y) / grad_steps  # scale loss when accumulating

            loss.backward()  # backpropagation: compute the gradients of the loss with respect to model parameters

            if (step + 1) % grad_steps == 0:
                optimizer.step()  # update the model parameters based on the gradients
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item())  # accumulate the loss for the current batch into the running loss

        # If dataloader size is not divisible by grad_steps, run final update
        if (len(dl_train) % grad_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        training_loss = running / max(len(dl_train), 1)  # calculate average training loss for the epoch

        # VALIDATION PHASE
        model.eval()  # setting the model to evaluation mode
        all_logits = []  # initialize a list to store all the logits for the validation set
        all_targets = []  # initialize a list to store all the targets for the validation set

        with torch.no_grad():  # disable gradient calculation for validation
            for x, y, _ in dl_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="mps", dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                all_logits.append(logits.detach().cpu())  # append the logits to the list moving to CPU
                all_targets.append(y.detach().cpu())  # append the targets to the list, moving to CPU

        logits = torch.cat(all_logits, dim=0)  # concatenate all the logits along the batch dimension
        targets = torch.cat(all_targets, dim=0)  # concatenate all the targets along the batch dimension

        metrics = compute_binary_metrics(_positive_logit(logits), targets)  # compute the binary classification metrics using logits and targets
        val_accuracy = metrics["accuracy"]  # extract the validation accuracy from the computed metrics

        # Some summary prints for the current epoch:
        print(f"[Fold {fold}, Epoch {epoch:02d}/{epochs}]")
        print(f"Training Loss: {training_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}, F1 Score: {metrics['f1_score']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")

        if metrics["f1_score"] > best_f1_score:
            best_f1_score = metrics["f1_score"]  # update the best F1 score
            best_val_accuracy = max(best_val_accuracy, val_accuracy)  # update the best validation accuracy
            torch.save(model.state_dict(), best_path)  # save the model state dict to the best path

        best_val_accuracy = max(best_val_accuracy, val_accuracy)  # update the best validation accuracy

        if stop_if_val_acc_perfect and val_accuracy >= 1 - perfect_acc_tolerance:
            print(f"Fold {fold} has reached perfect validation accuracy at epoch {epoch}. Stopping training to prevent overfitting!")
            break

        if device == "mps":
            torch.mps.empty_cache()

    summary = {
        "fold": fold,
        "best_f1_score": best_f1_score,
        "best_val_accuracy": best_val_accuracy,
        "best_model_path": best_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
