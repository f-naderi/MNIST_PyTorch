# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTCNN(nn.Module):
    """A simple convolutional neural network for MNIST"""
    def __init__(self, num_classes=10):
        super(SimpleMNISTCNN, self).__init__()
        
        # Convolutional block 1  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Convolutional block 3 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Flatten the feature maps
        self.flatten = nn.Flatten()
        
        # Fully connected layers  
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Fully connected layers  
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(device):
    """Create the model and move it to the appropriate device"""
    model = SimpleMNISTCNN(num_classes=10)
    model = model.to(device)
    return model