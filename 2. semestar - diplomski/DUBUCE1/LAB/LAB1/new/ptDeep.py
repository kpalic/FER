import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot, graph_surface, graph_data

class PTDeep(nn.Module):
    def __init__(self, layer_sizes, activation_function):
        """Arguments:
           - layer_sizes: list of integers defining number of neurons in each layer
           - activation_function: activation function for hidden layers (e.g., torch.relu)
        """
        super().__init__()
        self.activation_function = activation_function
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(nn.Parameter(torch.randn(layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(nn.Parameter(torch.randn(layer_sizes[i+1])))

    def forward(self, X):
        for i in range(len(self.weights) - 1):
            X = self.activation_function(torch.mm(X, self.weights[i]) + self.biases[i])
        logits = torch.mm(X, self.weights[-1]) + self.biases[-1]
        probs = torch.softmax(logits, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        probs = self.forward(X)
        # Dodavanje epsilona za numeričku stabilnost
        epsilon = 1e-12
        probs = torch.clamp(probs, epsilon, 1)
        loss = -torch.sum(Yoh_ * torch.log(probs)) / X.size(0)

        return loss

    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")
            total_params += param.numel()
        print(f"Total number of parameters: {total_params}\n")
        return total_params

def train(model, X, Yoh_, param_niter, param_delta):
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {i}, Loss: {loss.item()}')

def eval(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model.forward(X_tensor).detach().numpy()
    return probs

if __name__ == "__main__":
    # Inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    torch.manual_seed(100)

    # Generiraj podatke koristeći funkciju iz data.py
    X, Y_ = sample_gmm_2d(6, 2, 10) 

    # Pretvori oznake u one-hot notaciju koristeći funkciju iz data.py
    Yoh_ = class_to_onehot(Y_)

    # Pretvori u torch.Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Yoh_tensor = torch.tensor(Yoh_, dtype=torch.float32)

    # Definiraj model
    layer_sizes = [2, 10, 10, 2]
    ptd = PTDeep(layer_sizes, torch.relu)

    # Ispiši broj parametara
    ptd.count_params()

    # Nauči parametre
    train(ptd, X_tensor, Yoh_tensor, param_niter=10000, param_delta=0.1)

    # Dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptd, X)

    # Ispiši performansu
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == Y_)
    print(f'Accuracy: {accuracy}')

    # Iscrtaj rezultate koristeći funkcije iz data.py
    graph_surface(lambda x: eval(ptd, x)[:, 1], (X.min(axis=0), X.max(axis=0)))
    graph_data(X, Y_, preds)
    plt.show()
