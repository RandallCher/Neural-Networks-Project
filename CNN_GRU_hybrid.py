import torch
from torch import nn

class CNN_GRU(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_GRU, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.MaxPool1d(kernel_size=2),
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