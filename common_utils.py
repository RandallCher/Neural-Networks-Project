import numpy as np
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

#CNN functions
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        
        # Calculate the output size after the convolutional and pooling layers
        self.flatten = nn.Flatten()
        # The input length to the first linear layer needs to be calculated
        self.fc1 = nn.Linear(256 * self._get_conv_output_size(187), 256)  # Adjust this based on output size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output_size(self, input_length):
        # Calculate the output length after the conv/pool layers
        output_length = input_length
        
        # Each Conv1d reduces the length by (kernel_size - 1) on each side, then pooling reduces by half
        for _ in range(3):  # 3 convolutional layers
            output_length = (output_length - 2) // 2  # After conv and pooling

        return output_length

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + Pool
        x = self.flatten(x)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # Output layer
        return x
    
def evaluateCNN_model(x_data, y_label, num_classes, model, batch_size=32, device='cpu'):
    # Initialize Label Encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_label.values.ravel())
    y = np.eye(num_classes)[y]  # One-hot encoding

    # Prepare data for model
    X = x_data
    X_test = np.expand_dims(X, axis=1)    # Add a new dimension to match model input
    X_test_tensor = torch.tensor(X_test).float()    # Convert to float tensor
    y_test_tensor = torch.tensor(y).long()          # Convert to long tensor for labels

    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    # Evaluate accuracy
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to specified device
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels.argmax(dim=1)).sum().item()

    accuracy = correct_test / total_test  # Calculate test accuracy
    print(f'Test Accuracy for CNN model: {accuracy:.4f}')
    return accuracy

def set_and_get_seed(seed=42):
    '''
    Set random seeds
    
    Parameters
    ----------
    seed : int
        seed, default=42

    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def get_dataloader(X, y, is_train, batchSize=32):
    '''
    Get DataLoader object given X and y inputs

    Parameters
    ----------
    X : array-like

    y : array-like

    is_train : bool  
        sets the shuffle parameter in DataLoader class, true for train datasets, false for test datasets

    '''
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float), torch.as_tensor(y, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=is_train)

    return dataloader

def train(model, optimizer, train_loader, device, criterion=nn.CrossEntropyLoss()):
    '''
    One training epoch
    '''
    model.train()

    train_correct = 0
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.long())
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights

        # Get predictions
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == y_batch).sum().item()
        
        train_loss += loss.item() * X_batch.size(0)  # Accumulate loss

    # Calculate average training loss
    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    return train_loss, train_accuracy

def evaluate(model, test_loader, device, scheduler=None, criterion=nn.CrossEntropyLoss()):
    '''
    Function for validation/testing
    '''
    model.eval()

    test_loss = 0.0
    test_acc = 0.0
    correct = 0

    with torch.no_grad():  # No need to calculate gradients for validation/testing
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long())
            test_loss += loss.item() * X_batch.size(0)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
        
        # Calculate average validation loss and accuracy
        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        if scheduler != None:
            scheduler.step(test_loss)
        
    return test_loss, test_acc

def train_and_evaluate(model, optimizer, train_loader, val_loader, device, num_epochs, scheduler=None):
    model = model.to(device)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, train_loader, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = evaluate(model, scheduler, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Accuracy: {train_acc * 100:.2f}%, '
            f'Validation Loss: {val_loss:.4f}, '
            f'Validation Accuracy: {val_acc * 100:.2f}%')
        
    return train_losses, train_accuracies, val_losses, val_accuracies