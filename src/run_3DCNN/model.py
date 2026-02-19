import torch
import torch.nn as nn
from torchvision.models import r3d_18, R3D_18_Weights
from config import use_pretrained_model
#Baseline 3D CNN model using ResNet-18 architecture for video classification, adapted for our specific task of classifying normal vs. abnormal videos.
def build_model():
    if use_pretrained_model:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    else:
        model = r3d_18(weights=None)

    #Replacing the final connected layer to match the number of classes in our dataset
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # Assuming binary classification (e.g., normal vs. abnormal)
    return model