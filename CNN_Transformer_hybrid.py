import torch
from torch import nn
import torch.functional as F

class CNNTransformerHybrid(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, d_model=128):
        super(CNNTransformerHybrid, self).__init__()

        # CNN Feature extractor with Conv1d
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # MaxPooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Batch normalization after convolutions
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Projection layer to match transformer input size
        self.projector = nn.Linear(256, d_model)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_layers=num_layers
        )

        # Fully connected layer to output the final classification
        self.fc = nn.Linear(d_model, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is of shape (batch_size, channels, seq_len)
        if x.ndimension() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers with ReLU, batch normalization, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) 
        
        # Flattening and projection for transformer input
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        x = self.projector(x)  # Project to transformer input size

        # Transformer encoding
        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, d_model)
        x = self.encoder(x)  # Apply transformer layers

        # Pool the transformer output (use the last token for classification or apply mean)
        x = x.mean(dim=0)  # Aggregate over sequence length (or use last token: x[-1])
        x = self.fc(self.dropout(x))  # Fully connected layer after dropout

        return x