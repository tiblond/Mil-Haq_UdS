import merlin as ML # Package: merlinquantum, import: merlin
import torch
import data

# file_path = 'train.xlsx'
# X_raw, y_raw, dates = data.load_and_prepare_data(file_path, 1)


# Create a simple quantum layer
quantum_layer = ML.QuantumLayer.simple(
    input_size=3,
    n_params=50  # Number of trainable quantum parameters
)

# Use it like any PyTorch layer
x = torch.rand(10, 3)
output = quantum_layer(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")

import torch
import torch.nn as nn

# Merlin imports
import merlin.quantum as MQ 
from merlin.quantum.qpu_embedding import QPUEmbedding
from merlin.quantum.qonn import QONNLayer
from merlin.quantum.qpu_measure import QPUMeasurementLayer


class SmallLSTM_QML(nn.Module):
    def __init__(self, input_shape, hidden, num_layers, output_dim):
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_size = hidden
        self.num_layers = num_layers

        # --- Classical part (LSTM)
        self.lstm = nn.LSTM(self.input_shape,self.hidden_size,self.num_layers,batch_first=True)

        # --- Quantum part (Merlin)
        self.q_embed = QPUEmbedding(
            n_features=self.hidden_size,   # LSTM output is vector of size hidden
            n_modes=4                      # small photonic circuit
        )

        self.q_layer = QONNLayer(
            n_modes=4,
            depth=2                        # very small, very fast
        )

        self.q_measure = QPUMeasurementLayer(
            shots=100,                     # or None for exact expectation
            measurement="counts"           # returns (batch, n_modes)
        )

        # --- Classical output classifier
        self.linear = nn.Linear(4, self.output_dim)   # 4 = number of modes

    def forward(self, x):

        # Initial LSTM hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Classical LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the LAST time step
        lstm_out = out[:, -1, :]   # shape (batch, hidden_size)

        # ----- Quantum Block -----

        # Encode classical → quantum
        q_state = self.q_embed(lstm_out)

        # Apply photonic trainable unitary
        q_state = self.q_layer(q_state)

        # Measure → classical vector (batch, 4)
        q_out = self.q_measure(q_state)

        # ----- Final classifier -----
        out = self.linear(q_out)
        return out
