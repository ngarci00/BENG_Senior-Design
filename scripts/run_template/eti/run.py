# main() will run the ETI template example: load data, define model, train, evaluate.
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from .dataset import ETIDataset, collate_fn
from .model import Simple3DCNN
from .train import train_epoch, eval_model
from .config import  NUM_CLASSES, NUM_FRAMES, RESIZE_HEIGHT

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, if not use CPU
    print(f"Using device: {device}")

    # Get repository root and manifest path
    repo_root = Path(__file__).resolve().parents[2] # Get the root directory of the repository
    manifest = repo_root / "data" / "manifest.json" # Path to the data manifest file
    print(f"Using manifest at: {manifest}")
    print("Loading dataset...")
    
    # Load datasets
    train_ds = ETIDataset(manifest,"train",NUM_FRAMES, RESIZE_HEIGHT, True) # Load training dataset
    val_ds = ETIDataset(manifest,"val",NUM_FRAMES, RESIZE_HEIGHT, False) # Load validation dataset
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")# Print dataset sizes

    # Creating data loaders
    loader_train = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn) # Training data loader
    loader_val = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn) # Validation data loader

    # Initialize model, optimizer
    model = Simple3DCNN(NUM_CLASSES).to(device) # Initialize the 3D CNN model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Adam optimizer, learning rate 0.0001

    for epoch in range(5): # Train for 5 epochs
        train_loss = train_epoch(model, loader_train, optimizer, device) # Train for one epoch
        val_loss = eval_model(model, loader_val, device) # Evaluate on validation set
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}") # Print losses

