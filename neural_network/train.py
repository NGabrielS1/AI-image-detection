import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Class
class SiameseNeuralNetwork(nn.module):
    def __init__(self):
        super().__init__()
        # load ResNet50 (transfer learning)
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # change input and output
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    def forward(self, X1, X2):
        y1 = self.resnet50(X1)
        y2 = self.resnet50(X2)

        return y1, y2
