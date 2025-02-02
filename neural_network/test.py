import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import time

from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
import torch.nn.functional as F

# find device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# set random seed
random.seed(41)
torch.manual_seed(41)

# Function to create dataset and labels
class CreateDataset(Dataset):
    def __init__(self,imageFolderDataset):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        
    def __getitem__(self,index):
        img0_tuple = self.imageFolderDataset.imgs[index]

        # Determine if class is same or different
        class_type = random.randint(0,1)

        if class_type: # Find Image with same class 
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else: # Find Image with different class
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        # return 2 images + label        
        return img0, img1, torch.tensor([int(img1_tuple[1] != img0_tuple[1])], dtype=torch.float32)
    
    # len function
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# Model Class
class SiameseNetwork(nn.Module):
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

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive

# testing
if __name__ == "__main__":
    # load datasets
    test_dataset = CreateDataset(datasets.ImageFolder(root="./data/test/"))

    # create dataloaders
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=10)

    # load saved model
    model = SiameseNetwork()
    model.load_state_dict(torch.load("AI_DETECTOR_SIAMESE.pt", map_location=torch.device('cpu')))

    # variables
    correct = 0
    predictions = []

    # put model in eval mode
    with torch.no_grad():
      for b, (X1, X2, label) in enumerate(test_dataloader):
        # move to device
        X1, X2, label = X1.to(device), X2.to(device), label.to(device)
        # Get results from model
        y1, y2 = model(X1, X2)
        # get prediction
        prediction = F.pairwise_distance(y1, y2)
        print(f"prediction before: {prediction}")

        # Apply thresholding to each element
        prediction = (prediction > 1.5).float()
        print(f"prediction after: {prediction}")
        predictions.append(prediction)
        
        # Compare each item individually
        correct += (prediction == label).sum().item()

    print(f"{correct/len(predictions)}")  