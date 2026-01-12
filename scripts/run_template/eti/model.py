#Model Import
import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_outputs: int):
        """A simpple 3D CNN model for video classification.
        
        Args:
            num_opuuts(integer): Number of output classes for classification."""
        super().__init__()
        self.features = nn.Sequential(
        nn.Conv3d(3,32,3, stride = (1,2,2),padding=1), #Input channels=3 (RGB), output channels=32, kernel size=3, stride means how much the filter moves at each step
        nn.BatchNorm3d(32), # Batch normalization to stabilize training 
        nn.ReLU(), #Activation function
        nn.Conv3d(32,64,3, stride=2, padding=1), #Second conv layer, input channels=32, output channels=64, kernel size=3, stride=2, padding adds zeros around the input
        nn.BatchNorm3d(64), # Batch normalization to stabilize training
        nn.ReLU(), #Activation function
        nn.AdaptiveAvgPool3d((1,1,1)) #Adaptive average pooling to reduce spatial dimensions to 1x1x1   
    )
        self.fc = nn.Linear(64, num_outputs) #Fully connected layer to map features to output classes
    def forward(self,x):
        x = self.features(x) #Extracat features using convolutional layers
        return self.fc(x.flatten(1)) #Flatten the features and pass through the fully connected layer to get class scores
    