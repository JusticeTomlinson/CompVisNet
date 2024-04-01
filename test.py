import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def test_model(model, testloader, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # Since we're not training, we don't need to calculate gradients
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)   # Transfer data to the device
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = 100 * correct / total
    print(f'Average test loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

# To use this function, you need a testloader (as defined in the train_model function) and a trained model.
# Example usage (assuming the testloader and a model are already defined and the model is trained):
# test_model(model, testloader)
