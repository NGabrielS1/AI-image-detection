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

# load datasets
train_dataset = CreateDataset(datasets.ImageFolder(root="./dataset/train/"))
test_dataset = CreateDataset(datasets.ImageFolder(root="./dataset/test/"))

# create dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=10)
test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=10)

# Model Class
class SiameseNetwork(nn.module):
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

# create a instance of model, choose loss function and optimizer
model = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

# variables
epochs = 5
train_losses = []
test_losses = []

# Loop of Epochs
for i in range(epochs):
    # train
    for b, (X1, X2, label) in enumerate(train_dataloader, 0):
        # Zero the gradients
        optimizer.zero_grad()
        # Get results from model
        y1, y2 = model(X1, X2)
        # Pass results and label to loss function
        loss = criterion(y1, y2, label)
        # Calculate backpropagation and optimize
        loss.backward()
        optimizer.step()

        # print loss
        if b % 100 == 0:
            print(f"Epoch: {i}, Batch: {b}, Loss: {loss.item()}")
    
    # track loss each epoch
    train_losses.append(loss.item())
