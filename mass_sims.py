from train import train_model
from test import test_model
from conv_config import one_conv_configs, two_layer_configs, three_layer_configs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader 


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def get_cifar10_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 images are already 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def count_parameters_detailed(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Number of Parameters: {param.numel()}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

class OneLayerCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, num_classes, output_size):
        super(OneLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Assuming an output_size of 16 after pooling for a 32x32 input image
        self.fc = nn.Linear(out_channels * output_size * output_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class TwoLayerCNN(nn.Module):
    def __init__(self, in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                 in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2, num_classes, output_size):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1,
                               stride=stride1, padding=padding1, dilation=dilation1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels2, kernel_size=kernel_size2,
                               stride=stride2, padding=padding2, dilation=dilation2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Assuming an output_size of 8 after two pooling layers for a 32x32 input image
        self.fc = nn.Linear(out_channels2 * output_size * output_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ThreeLayerCNN(nn.Module):
    def __init__(self, in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                 in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
                 in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3, num_classes):
        super(ThreeLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1,  # Adjusted for RGB input
                               stride=stride1, padding=padding1, dilation=dilation1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels2, kernel_size=kernel_size2,
                               stride=stride2, padding=padding2, dilation=dilation2)
        self.conv3 = nn.Conv2d(in_channels3, out_channels3, kernel_size=kernel_size3,
                               stride=stride3, padding=padding3, dilation=dilation3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_classes = num_classes
        
        # Placeholder for the FC layer, will be replaced
        self.fc = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
            
            # Dynamically determine the size for the fully connected layer
        if self.fc is None:
            num_features = x.view(x.size(0), -1).shape[1]
            self.fc = nn.Linear(num_features, self.num_classes).to(x.device)  # Ensure the layer is on the same device as input

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

def mass_test_cnn_models(trainloader, testloader, conv_configs, num_classes, num_epochs=50):
    results = []
    for config in conv_configs:

      in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1, \
      in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2, \
      in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3 = config
            
      model = ThreeLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                                  in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
                                  in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3,
                                  num_classes)
      count_parameters_detailed(model)
            
      print(f"Training model-- conv_config={config}")
      train_model(model, trainloader, epochs=num_epochs)
            
      print("Testing model...")
      accuracy = test_model(model, testloader)  
            
        # Collect the results
      results.append((config, accuracy))
    
    return results


# fc_sizes = [64, 128, 256] # may want to make this variable too
trainloader, testloader = get_cifar10_loaders(batch_size=64)
results = mass_test_cnn_models(trainloader, testloader, three_layer_configs, 10, num_epochs=50)
