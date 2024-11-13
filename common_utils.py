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
        
        self.flatten = nn.Flatten()
       
        self.fc1 = nn.Linear(256 * self._get_conv_output_size(187), 256)  # Adjust this based on output size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output_size(self, input_length):

        output_length = input_length
        
        # 3 convolutional layers
        for _ in range(3):  
            output_length = (output_length - 2) // 2  

        return output_length

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))  
        x = self.flatten(x)  
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)  
        x = self.fc2(x)  
        return x
    
def evaluateCNN_model(x_data, y_label, num_classes, model, batch_size=32, device='cpu'):

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_label.values.ravel())
    y = np.eye(num_classes)[y]  


    X = x_data
    X_test = np.expand_dims(X, axis=1)   
    X_test_tensor = torch.tensor(X_test).float()  
    y_test_tensor = torch.tensor(y).long()        

  
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()


    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels.argmax(dim=1)).sum().item()

    accuracy = correct_test / total_test  
    print(f'Test Accuracy for CNN model: {accuracy:.4f}')
    return accuracy

#RNN Model
class GRUModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, numClasses):
        super(GRUModel, self).__init__()
        
        # Define the GRU layer
        self.gru = nn.GRU(input_size=inputSize, 
                          hidden_size=hiddenSize, 
                          num_layers=numLayers, 
                          batch_first=True
                          )
        
        # Define a fully connected output layer
        self.fc = nn.Linear(hiddenSize, numClasses)
    
    def forward(self, x):
        # Initialize hidden state for GRU
        h0 = torch.zeros(self.gru.num_layers * (2 if self.gru.bidirectional else 1), x.size(0), self.gru.hidden_size).to(x.device)
        
        # Forward propagate through GRU
        out, _ = self.gru(x, h0)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        return out
# Transformers evaluation

def cnn_transformer_evaluate(model, test_loader, criterion, device):
    model.eval()  

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No need to compute gradients during testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)         
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# CNN_GRU
class CNN_GRU(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_GRU, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(num_features=256),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
        )

        self.gru = nn.GRU(input_size=256, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.permute(x, (0, 2, 1))

        x, hidden_state = self.gru(x)
        x = x[:, -1, :]

        x = self.fc(x)
        
        return x


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

def evaluate(model, test_loader, device, criterion=nn.CrossEntropyLoss()):
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
        
    return test_loss, test_acc

def train_and_evaluate(model, optimizer, train_loader, val_loader, device, num_epochs, scheduler=None):
    model = model.to(device)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, train_loader, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = evaluate(model, val_loader, device)
        test_losses.append(test_loss)
        test_losses.append(test_acc)

        if scheduler:
            scheduler.step()

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Accuracy: {train_acc * 100:.2f}%, '
            f'Test Loss: {test_loss:.4f}, '
            f'Test Accuracy: {test_acc * 100:.2f}%')
        
    return train_losses, train_accuracies, test_losses, test_accuracies