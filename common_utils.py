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

import os
import copy
import pandas as pd
import math

from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data.dataset import Dataset
from torch.optim import Adam

def load_list(root, filename):
    filepath = os.path.join(root, filename)
    output = pd.read_csv(filepath, header=None)
    return output


class MITBIHArrhythmia(Dataset):
    def __init__(self, root, subset=None):
        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` is not None, it must be one of {'training', 'validation', 'testing'}."
        )
        self.root = root
        file_dict = {
            "training": "mitbih_train.csv",
            "validation": "mitbih_train.csv",  # Assuming same file for both test and validation
            "testing": "mitbih_test.csv"
        }
        
        if subset is None:
            raise ValueError("Subset must be specified as 'training', 'validation', or 'testing'")
        
        # Load the data file as a DataFrame
        self._walker = load_list(self.root, file_dict[subset])
        
    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n: int):
        # Access the nth row, convert it to a list, and split data from label
        row = self._walker.loc[n, :].values.tolist()
        label = row.pop()  # Last element is assumed to be the label
        return row, label
    
class LitMITBIH(LightningDataModule):
    def __init__(self, root, batch_size, num_workers, length=200):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = root
        self.length = length
    
    def prepare_data(self):
        self.train_dataset = MITBIHArrhythmia(self.path, "training")
        self.val_dataset = MITBIHArrhythmia(self.path, "validation")
        self.test_dataset = MITBIHArrhythmia(self.path, "testing")
    
    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        labels = []
        heartbeats = []
        for sample in batch:
            waveform, label = sample
            if len(waveform) < self.length:
                padsize = self.length - len(waveform)
                waveform += [0]*padsize

            labels.append(torch.tensor(label).type(torch.int64))
            heartbeats.append(torch.tensor(waveform))

        labels = torch.stack(labels)
        heartbeats = torch.stack(heartbeats)
        return heartbeats, labels
    
from torchmetrics import Accuracy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#### ATTENTION
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all h heads.

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view( -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2).contiguous().view( -1, self.h * self.d_k))
        
        del query
        del key
        del value
        out = self.linears[-1](x)
        return out

## BLOCKING
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class FeedForward(nn.Module):
    "Construct a FeedForward network with one hidden layer"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class Transformer(nn.Module):
    "Transformer Model"
    def __init__(self, input_size, num_classes, num_heads=8, N=6, d_ff=256, dropout=0.0):
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, input_size)
        ff = FeedForward(input_size, d_ff, dropout)
        self.encoder = Encoder(EncoderBlock(input_size, c(attn), c(ff), dropout), N)
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class LitTransformer(LightningModule):
    def __init__(self, input_size, num_classes, num_heads, depth, max_epochs, lr, dropout=0.1, d_ff=256):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(input_size, num_classes, num_heads, depth, d_ff, dropout)
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer  # Add scheduler if needed

    def training_step(self, batch, batch_idx):
        wavs, labels = batch
        preds = self(wavs)
        loss = self.loss(preds, labels)
        
        # Log training loss
        self.log('train_loss', loss)

        # Log accuracy using the class instance
        acc = self.accuracy(preds.softmax(dim=-1), labels) * 100.0
        self.log('train_acc', acc)

        return {"loss": loss}
    
    def on_training_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.trainer.callback_metrics]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        wavs, labels = batch
        preds = self(wavs)
        loss = self.loss(preds, labels)
        
        # Log validation loss and accuracy
        self.log('val_loss', loss)
        acc = self.accuracy(preds.softmax(dim=-1), labels) * 100.0
        self.log('val_acc', acc)

        return {"preds": preds, 'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([self.trainer.callback_metrics['val_loss']]).mean()
        avg_acc = torch.stack([self.trainer.callback_metrics['val_acc']]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", avg_acc, on_epoch=True, prog_bar=True)



    def test_step(self, batch, batch_idx):
        wavs, labels = batch
        preds = self(wavs)
        loss = self.loss(preds, labels)
        acc = self.accuracy(preds.softmax(dim=-1), labels) * 100.0
        self.log('test_acc', acc, prog_bar=True)
        return {"preds": preds, 'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)
