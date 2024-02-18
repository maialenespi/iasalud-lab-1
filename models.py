import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6000, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)

        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 6000)
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLP, self).__init__()

        self.fc_input = nn.Linear(input_size[1], hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.activation(self.fc_input(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        #x = self.fc_output(x)
        return x
    

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size, stride, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_blocks.append(self._make_conv_block(1, hidden_size, kernel_size, stride))
                continue
            else:
                self.conv_blocks.append(self._make_conv_block(hidden_size, hidden_size // 2, kernel_size, stride))
            if hidden_size >= 2:
                hidden_size = hidden_size // 2
            else:
                hidden_size = 2
        self.fc1 = nn.Linear(self._get_conv_output(input_size), 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = 3),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(out_channels)
        )

    def _get_conv_output(self, input_size):
        x = torch.randn(input_size).unsqueeze(1)
        for block in self.conv_blocks:
            x = block(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x