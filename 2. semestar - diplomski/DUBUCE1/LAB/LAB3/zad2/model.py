# model.py
import torch.nn as nn
import torch

# Define the Baseline Model
class BaselineModel(nn.Module):
    def __init__(self, embedding_layer):
        super(BaselineModel, self).__init__()
        self.embedding = embedding_layer
        
        embedding_dim = 300
        self.fc1 = nn.Linear(embedding_dim, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # mean pooling

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x

# Function to initialize the model
def initialize_model(embedding_layer, args):
    model = BaselineModel(embedding_layer).to(args.device)
    return model
