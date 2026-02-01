# Here is where the actual training process starts for our model.
import json
import torch
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.utils.data as DataLoader
from dataset import VideoClipDataset

#Defining the Model
def build_3DCNN(num_classes: int =1)-> nn.Module:
    """Builds a simple 3D CNN model for video classification."""
    try:
        import torchvision
        from torchvision.models.video import r3d_18
        model = None
        try:
            weights = torchvision.models.video.R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
        except Exception: 
            try:
                model = r3d_18(pretrained=True)
            except Exception:
                model = r3d_18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    except Exception as e:
        raise RuntimeError("Error importing torchvision or building the model") from e
   
        raise