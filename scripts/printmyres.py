#This script serves as a simple utility to display the resized aspect ratios before training the model.
#We are comparing: 64x64, 128x128, 224x224, 320x320, 600x600
import os
from src.run_SVM.config import resize_hw

#load the image preferabely one with all the annotations present, and display the resized aspect ratio
