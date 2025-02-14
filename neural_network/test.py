import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import time
from statistics import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchvision.utils import make_grid
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet34_Weights
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
        # data transformation and augmentation    
        self.transform = transforms.Compose([
            transforms.Resize((100,100)),
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
        # load ResNet34(transfer learning)
        self.resnet34 = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet34.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    
    def forward_once(self, X):
        y = self.resnet34(X)

        return y

    def forward(self, X1, X2):
        y1 = self.forward_once(X1)
        y2 = self.forward_once(X2)

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
    
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(("neural_network/AI_DETECTOR_SIAMESE.pt"),map_location=torch.device('cpu')))

if __name__ == "__main__":
    # load datasets
    train_dataset = CreateDataset(datasets.ImageFolder(root="./data/train/"))
    test_dataset = CreateDataset(datasets.ImageFolder(root="./data/test/"))

    # create dataloaders
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=1)
    
    # variables
    correct = 0
    y_test = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for b, (x0, x1, label) in enumerate(test_dataloader):
            predlabel = 0
            # Concatenate the two images together
            concatenated = torch.cat((x0, x1), 0)

            # move to device
            x0, x1, label = x0.to(device), x1.to(device), label.to(device)
            
            output1, output2 = model(x0, x1)
            euclidean_distance = F.pairwise_distance(output1, output2)
            # imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
            if euclidean_distance.item() < 1:
                predlabel = 0.0
            else:
                predlabel = 1.0
            if predlabel == label.item():
                correct += 1
            
            # create list of predictions and labels
            y_test.append(label.item())
            y_pred.append(predlabel)
            
            if b % 1000 == 0:
                print(b)
        
        print(b+1)
        # find accuracy
        print(f"Accuracy: {correct/(b+1)*100}%")

        # create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Same", "Different"]).plot()