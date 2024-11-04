import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        
        # Update this based on your input size after the conv/pooling layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2, 256)  # Calculate output_length based on your input
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CNN1D_2(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D_2, self).__init__()
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