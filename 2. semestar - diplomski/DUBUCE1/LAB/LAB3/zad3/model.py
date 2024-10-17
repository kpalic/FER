# model.py
import torch.nn as nn
import torch

# Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=150, num_layers=2):
        super(GRUModel, self).__init__()
        self.embedding = embedding_matrix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=embedding_matrix.embedding_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take the output of the last time step

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        return x

# Function to initialize the model
def initialize_model(embedding_matrix, args):
    model = GRUModel(embedding_matrix).to(args.device)
    return model
