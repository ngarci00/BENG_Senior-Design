#Labels, constants, and utilities for ETI video classification
from typing import Dict, List #For type hinting
import torch #PyTorch library

# Label map for video classification
DEFAULT_LABELS: Dict[str,int] = { "background": 0,
                                  "vocal_cords": 1,
                                    "epiglottis": 2, 
                                    "arytenoids": 3,
                                      "esophagus": 4, 
                                      "endotracheal_tube": 5 } #These are the data labels we want to stick to <-
#Exclude background from the number of classes
CLIP_LABELS: List[str] = [ name for name, idx in sorted(DEFAULT_LABELS.items(), key=lambda kv: kv[1])
    if idx != 0
]# List of labels excluding background
NUM_CLASSES = len(CLIP_LABELS) #Number of classes excluding background

NUM_FRAMES = 16 #Number of frames per video clip
RESIZE_HEIGHT =  (112,112) #Height to resize video frames

def dict_to_tensor(presence:dict) -> torch.Tensor:
    """We are converting a dictionary of label presence into a tensor format for model training."""
    y = torch.zeros(NUM_CLASSES, dtype=torch.float32) #Initialize a tensor of zeros with size equal to number of classes
    for i, label in enumerate(CLIP_LABELS): #Iterate over the labels
        y[i] = float(bool(presence.get(label, False))) #Set tensor value to 1.0 if label is present, else 0.0
    return y #Return the tensor
