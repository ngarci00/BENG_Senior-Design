#training script that evaluates the model
import torch
import torch.nn as nn
def train_epoch(model, loader, optimizer, device):
    """ Trains the model for one epoch!
    Args:
        model: The neural network model to be trained.
        loader: DataLoader providing the training data.
        optimizer: The optimizer used for updating model weights.
        device: The device (CPU or GPU) to perform computations on."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()#Binary Cross Entropy Loss with Logits
    total = 0.0

    for x, y in loader: #Iterate over data loader
        x, y = x.to(device), y.to(device) 
        optimizer.zero_grad()#Zero the gradients
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad() #No gradient computation for evaluation
def eval_model(model, loader, device):
    """ Evaluates the model on the validation/test set!
    Args:
        model: The neural network model to be evaluated.
        loader: DataLoader providing the evaluation data.
        device: The device (CPU or GPU) to perform computations on."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += criterion(model(x), y).item()
    return total / len(loader)
