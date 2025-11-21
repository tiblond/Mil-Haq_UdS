import torch
import torch.nn as nn
# Merlin imports
import merlin as ML


class Q_LSTM(nn.Module):
    def __init__(self, input_shape, hidden, num_layers, output_dim):
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_size = hidden
        self.num_layers = num_layers

        # --- Classical part (LSTM)
        self.lstm = nn.LSTM(self.input_shape,self.hidden_size,self.num_layers,batch_first=True)

        # --- Quantum Part
        self.builder = ML.CircuitBuilder(n_modes=4)
        self.builder.add_superpositions(depth=1)                 # entangle all qubits together
        self.builder.add_angle_encoding(modes=[0, 1], name="x")  # encode the classical info into the q2uantum cirquit
        self.builder.add_rotations(trainable=True, name="theta") # this is a quantum layer
        self.builder.add_superpositions(depth=1, trainable=True)

        self.Qlayer = ML.QuantumLayer(
            input_size=2,
            builder=self.builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            no_bunching=True,
        )

        x = torch.rand(4, 2)
        probs = layer(x)

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
        q_out = self.Qlayer(lstm_out)

        # ----- Final classifier -----
        out = self.linear(q_out)
        return out