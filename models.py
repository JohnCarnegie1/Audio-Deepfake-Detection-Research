import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron
from torchvision import models

class SpikingCNN(nn.Module):
    def __init__(self):
        super(SpikingCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,5,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,5,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,5,padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.lif1 = neuron.LIFNode(v_threshold=0.07, v_reset=0.0, tau=2.0)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128,64)
        self.dropout = nn.Dropout(0.1)
        self.lif2 = neuron.LIFNode()
        self.fc2 = nn.Linear(64,2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.lif1(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.lif2(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(SpikingCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)   # Input: [B, 100, 256, 256]
        self.pool1 = nn.MaxPool2d(2, 2)                           # Output: [B, 32, 128, 128]

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)                           # Output: [B, 64, 64, 64]

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)                           # Output: [B, 128, 32, 32]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))           # Output: [B, 128, 1, 1]
        self.flatten = nn.Flatten()                               # Output: [B, 128]

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.2) 
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        
        x = self.dropout(x)  

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ResNet50Binary(nn.Module):
    def __init__(self):
        super(ResNet50Binary, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        
        # Replace final fully connected layer for binary classification
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.base_model(x)

class ResNet50Spiking(nn.Module):
    def __init__(self):
        super(ResNet50Spiking, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        # Removing the last FC layer
        self.base_model.fc = nn.Identity()
        self.lif1 = neuron.LIFNode(v_threshold=0.03, tau=5.0)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2048, 128)
        self.lif2 = neuron.LIFNode()
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.base_model(x)     # ResNet feature extractor
        x = self.lif1(x)           # Spiking layer 1
        x = self.dropout(x)
        x = self.lif2(self.fc1(x)) # Spiking layer 2
        x = self.fc2(x)
        return x
#TODO eficientNet with and without LIF node

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = models.efficientnet_b7(pretrained=True)
        num_features = self.base_model.classifier[1].in_features

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.base_model(x)

class EfficientNetSpiking(nn.Module):
    def __init__(self):
        super(EfficientNetSpiking, self).__init__()
        self.base_model = models.efficientnet_b7(pretrained=True)

        # Remove the final classification layer to get feature embeddings
        self.base_model.classifier = nn.Identity()

        # Spiking neuron after the feature extractor
        self.lif1 = neuron.LIFNode(v_threshold = 0.03, tau = 5.0)

        # Dropout + fully connected layers for binary classification
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2560, 128)
        self.lif2 = neuron.LIFNode()
        self.fc2 = nn.Linear(128, 2) # Binary Classification

    def forward(self, x):
        x = self.base_model(x) # EfficientNet feature extractor
        x = self.lif1(x) # Spiking layer 1
        x = self.dropout(x)
        x = self.lif2(self.fc1(x)) # Spiking layer 2
        x = self.fc2(x)
        return x