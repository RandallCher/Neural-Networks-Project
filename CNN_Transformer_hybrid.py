import torch
from torch import nn
import torch.nn.functional as F

class CNNTransformerHybrid(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, d_model=128):
        super(CNNTransformerHybrid, self).__init__()

        # CNN Feature extractor with Conv1d
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.projector = nn.Linear(256, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.ndimension() == 2:
            x = x.unsqueeze(1)  

        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) 

        x = x.permute(0, 2, 1)  
        x = self.projector(x)  

        # Transformer encoding
        x = x.permute(1, 0, 2) 
        x = self.encoder(x) 


        x = x.mean(dim=0) 
        x = self.fc(self.dropout(x)) 

        return x


# Model instantiation
