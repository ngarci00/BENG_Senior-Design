import torch, torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from config import use_pretrained_model, backbone
#Baseline 2D CNN model using ResNet-18 architecture for video classification, adapted for our specific task of classifying normal vs. abnormal videos.
class FrameAveraged2DCNN(nn.Module):
    def __init__(self,backbone:str = "resnet18", pretrained: bool = True, num_classes: int =2):
        super().__init__()

        bb = (backbone or "resnet18").lower()
        if bb == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.net = resnet34(weights=weights)
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.net = resnet18(weights=weights)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input tensor to have 5 dimensions (B, C, T, H, W), but got {tuple(x.shape)} dimensions.")
        B, C, K, H, W = x.shape
        x =x.view(B * K , C, H, W)  #Reshape to (B*K, C, H, W) to process each frame independently through the 2D CNN
        logits = self.net(x)  #Shape: (B*K, num_classes
        logits = logits.view(B, K, -1)  #Reshape back to (B, K, num_classes)
        logits = logits.mean(dim=1)  # verage the logits across the K frames to get a single prediction per video
        return logits

def build_model() -> nn.Module:
    
    return FrameAveraged2DCNN(backbone=backbone, pretrained=bool(use_pretrained_model), num_classes=2)