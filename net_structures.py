# Create Variable Neural Network Architectures - DONE
# Create training paradigm 
# Load and begin testing the dataset
# create test training loop

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, Omniglot, CIFAR10
from torch.utils.data import DataLoader

class OneLayerCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, num_classes):
        super(OneLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(out_channels * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class TwoLayerCNN(nn.Module):
    def __init__(self, in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                 in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2, num_classes):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1,
                               stride=stride1, padding=padding1, dilation=dilation1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels2, kernel_size=kernel_size2,
                               stride=stride2, padding=padding2, dilation=dilation2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(8 * 8 * out_channels2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ThreeLayerCNN(nn.Module):
    def __init__(self, in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                 in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
                 in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3, num_classes):
        super(ThreeLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1,
                               stride=stride1, padding=padding1, dilation=dilation1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels2, kernel_size=kernel_size2,
                               stride=stride2, padding=padding2, dilation=dilation2)
        self.conv3 = nn.Conv2d(in_channels3, out_channels3, kernel_size=kernel_size3,
                               stride=stride3, padding=padding3, dilation=dilation3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(4 * 4 * out_channels3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x