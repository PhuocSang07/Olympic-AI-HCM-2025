import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
    
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b3(pretrained=True)
        self.features = self.model.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  
        # Flatten + dropout + linear
        in_features = self.model.classifier[1].in_features
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)                        
        x = self.adaptive_pool(x)                    
        x = torch.flatten(x, 1)                      
        x = self.dropout(x)                         
        x = self.fc(x)                               
        return x


class CustomTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_paths = sorted([os.path.join(test_dir, fname)
                                   for fname in os.listdir(test_dir)
                                   if fname.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)
    
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]       
        if self.transform:
            image = self.transform(image)     
        return image, label