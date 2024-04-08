import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import csv

import torch.optim as optim
import matplotlib.pyplot as plt
import csv

def train_model(model, trainloader, device, epochs, model_size, learning_rate=0.001, model_type='CNN', csv_file='training_metrics.csv'):
    model.train()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Ensure to write header for the CSV file if it's the first time
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        file.seek(0, 2)  
        if file.tell() == 0:
            writer.writerow(['Model Type', 'Epoch', 'Model Size', 'Total Parameters', 'Accuracy', 'Loss'])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        
        # Append epoch results to CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_type, epoch + 1, model_size, sum(p.numel() for p in model.parameters()), accuracy, avg_loss])

        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
        
    print('Finished Training')


    # # Plotting the training loss and accuracy
    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # color = 'tab:red'
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss', color=color)
    # ax1.plot(range(1, epochs+1), losses, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    # ax2.plot(range(1, epochs+1), accuracies, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title("Training Loss and Accuracy")
    # plt.show()

# Example usage:
# model = OneLayerCNN(...) or model = TwoLayerCNN(...) or model = ThreeLayerCNN(...)
# Make sure to define the models with the correct parameters before using them here.
# trainloader, testloader = ...  # Define your training and test data loaders
# train_model(model, trainloader, testloader)