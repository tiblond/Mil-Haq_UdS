import torch
import torch.nn as nn

class smallLSTM(nn.Module):
    def __init__(self, input_shape, hidden, num_layers, output_dim):
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_size = hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_shape, self.hidden_size, self.num_layers,  batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.linear(out[:, -1, :])
        return out