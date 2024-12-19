import torch
import torch.nn as nn

from config.config import config

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(config.ffnn.input_size, config.ffnn.input_size)
        self.fc2 = nn.Linear(config.ffnn.input_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x