import torch
import torch.nn as nn
from config.config import config


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=config.rnn.input_size,
            hidden_size=config.rnn.hidden_size,
            num_layers=config.rnn.num_layers,
            bias=config.rnn.bias,
            dropout=config.rnn.dropout,
            bidirectional=config.rnn.bidirectional,
            batch_first=True
        )
        fc_dim = config.rnn.hidden_size * 2 if config.rnn.bidirectional else config.rnn.hidden_size
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        classes = torch.relu(self.fc1(last_hidden))
        classes = self.fc2(classes)
        return (classes, out)
