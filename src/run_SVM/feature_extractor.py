#Since we can't really train a SVM with video data directly, we will extract features from the videos using a pre-trained 2D CNN (e.g., ResNet-18) and then train a SVM on those features. This script will handle the feature extraction part.
import torch, torch.nn as nn
from typing import Tuple
from torchvision import models

class ResNet18Embedder(nn.Module):
    """Example input: x [ N, 3, H, W ] 
    Output: z [N, 512] """
    def __init__(self,pretrained:bool=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) #Remove the final fully connected layer, we only want the features

    @torch.no_grad() #We don't need gradients for feature extraction
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x) #Extract features using the ResNet-18 backbone
        z = z.flatten(1) #Flatten the output to get a feature vector of shape [N, 512] 
        return z