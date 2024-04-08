from train import train_model
from test import test_model
from conv_config_cifar import one_layer_configs, two_layer_configs, three_layer_configs
from conv_config_mnist import one_layer_mnist, two_layer_mnist, three_layer_mnist
from net_structures import OneLayerCNN, TwoLayerCNN, ThreeLayerCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Number of Parameters: {param.numel()}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


def mass_test_cnn_models(trainloader, testloader, one_layer_configs, two_layer_configs, three_layer_configs, num_classes, num_epochs):
    one_layer_results, two_layer_results, three_layer_results = [], [], []

    print("Now Training One Layer Models")
    for config in one_layer_configs:
        model_type = '1 hidden'
        in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1 = config
            
        model = OneLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                                    num_classes)
        model.to(device)
        count_parameters(model)
        print(f"Training model: conv_config={config}")
        train_model(model, trainloader, device, num_epochs, config, model_type=model_type)
            
        print("Testing model...")
        accuracy = test_model(model, testloader, device, config, model_type=model_type)  
            
        one_layer_results.append((config, accuracy))

    print("Now Training Two Layer Models")
    for config in two_layer_configs:
        model_type = '2 hidden'

        in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1, out_channels2, kernel_size2, stride2, padding2, dilation2 = config

        model = TwoLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                        out_channels1, out_channels2, kernel_size2, stride2, padding2, dilation2,
                        num_classes)
        model.to(device)
        count_parameters(model)
        print(f"Training model: conv_config={config}")
        train_model(model, trainloader, device, num_epochs, config, model_type=model_type)
            
        print("Testing model...")
        accuracy = test_model(model, testloader, device, config, model_type=model_type)  
            
        two_layer_results.append((config, accuracy))

    print("Now Training Three Layer Models")
    for config in three_layer_configs:
        model_type = '3 hidden'

        in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1, \
        in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2, \
        in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3 = config
            
        model = ThreeLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                                    in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
                                    in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3,
                                    num_classes)
        model.to(device)
        count_parameters(model)
        print(f"Training model: conv_config={config}")
        train_model(model, trainloader, device, num_epochs, config, model_type=model_type)
            
        print("Testing model...")
        accuracy = test_model(model, testloader, device, config, model_type=model_type)  
  
            
        three_layer_results.append((config, accuracy))
    
    return one_layer_results, two_layer_results, three_layer_results


# trainloader, testloader = get_mnist_loaders(batch_size=64)
trainloader, testloader = get_cifar10_loaders(batch_size=64)

results_cifar = mass_test_cnn_models(trainloader, testloader, one_layer_configs, two_layer_configs, three_layer_configs, num_classes=10, num_epochs=5)
# results_mnist = mass_test_cnn_models(trainloader, testloader, one_layer_mnist, two_layer_mnist, three_layer_mnist, num_classes=10, num_epochs=5)
