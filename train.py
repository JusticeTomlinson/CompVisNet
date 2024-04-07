import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train_model(model, trainloader, epochs=50, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, momentum = 0.9)

    losses = []  # To store average loss per epoch
    accuracies = []  # To store accuracy per epoch

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate and print average loss per epoch
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.3f}')
        losses.append(avg_loss)  # Append average loss for plotting

        # Calculate accuracy after each epoch
        correct = 0
        total = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Accuracy: {accuracy:.2f}%')

    print('Finished Training')

    # Plotting the training loss and accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, epochs+1), losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(1, epochs+1), accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Training Loss and Accuracy")
    plt.show()

# different optimizer -- adam, different lr = 0.01, data augmentation, more epochs