import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import csv

def test_model(model, testloader, device, model_size, model_type, csv_file='test_results.csv'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = 100 * correct / total

    print(f'Average test loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        file.seek(0, 2) 
        if file.tell() == 0:
            writer.writerow(['Model Size', 'Model Type', 'Average Test Loss', 'Accuracy'])
        writer.writerow([model_size, model_type, avg_loss, accuracy])

    return avg_loss, accuracy

# To use this function, you need a testloader (as defined in the train_model function) and a trained model.
# Example usage (assuming the testloader and a model are already defined and the model is trained):
# test_model(model, testloader)