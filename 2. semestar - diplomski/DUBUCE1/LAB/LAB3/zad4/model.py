# model.py

import torch.nn as nn
import torch

class RNNModel(nn.Module):
    def __init__(self, embedding_layer, rnn_type='GRU', hidden_size=150, num_layers=2, dropout=0.0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.embedding = embedding_layer
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[rnn_type]
        self.rnn = rnn_class(input_size=embedding_layer.embedding_dim,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout if num_layers > 1 else 0.0,
                             bidirectional=bidirectional,
                             batch_first=True)
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Use the last hidden state
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def initialize_model(embedding_layer, args):
    model = RNNModel(embedding_layer, 
                     rnn_type=args.rnn_type, 
                     hidden_size=args.hidden_size, 
                     num_layers=args.num_layers, 
                     dropout=args.dropout, 
                     bidirectional=args.bidirectional).to(args.device)
    return model
