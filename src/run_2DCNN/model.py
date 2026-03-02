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
            raise RuntimeError(f"Expected input shape [B,K,C,H,W], got {tuple(x.shape)}")

        B, K, C, H, W = x.shape

        #If someone accidentally passes [B, C, K, H, W], auto-fix it<- fallback
        if C not in (1, 3) and x.shape[1] in (1, 3):
            # x is likely [B, C, K, H, W] -> convert to [B, K, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            B, K, C, H, W = x.shape

        if C != 3:
            raise RuntimeError(f"Expected C = 3 channels, got input shape {tuple(x.shape)}")

        x = x.view(B * K, C, H, W)  # [B*K, C, H, W]
        logits = self.net(x)        # [B*K, num_classes]
        logits = logits.view(B, K, -1).mean(dim=1)  # [B, num_classes]
        return logits

def build_model() -> nn.Module:
    
    return FrameAveraged2DCNN(backbone=backbone, pretrained=bool(use_pretrained_model), num_classes=2)