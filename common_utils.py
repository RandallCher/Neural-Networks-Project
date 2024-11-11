import numpy as np
import torch

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

# Transformers functions



import random

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

def get_dataloader(X, y, is_test, batchSize=32):
    '''
    Get DataLoader object given X and y inputs
    
    Parameters
    ----------
    X : array-like

    y : array-like

    is_test : bool  
        sets the shuffle parameter in DataLoader class, true for train datasets, false for test datasets

    '''
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float), torch.as_tensor(y, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=is_test)

    return dataloader