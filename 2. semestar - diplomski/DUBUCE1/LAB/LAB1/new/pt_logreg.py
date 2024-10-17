import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, graph_surface, graph_data, class_to_onehot

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.randn(C))

    def forward(self, X):
        # Unaprijedni prolaz modela: izračunati vjerojatnosti
        logits = torch.mm(X, self.W) + self.b
        probs = torch.softmax(logits, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        # Formulacija gubitka
        epsilon = 1e-12
        probs = self.forward(X)
        loss = -torch.sum(Yoh_ * torch.log(probs + epsilon)) / X.size(0)
        return loss

def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
    - X: model inputs [NxD], type: torch.Tensor
    - Yoh_: ground truth [NxC], type: torch.Tensor
    - param_niter: number of training iterations
    - param_delta: learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    for i in range(param_niter):
        # Zero grad
        optimizer.zero_grad()

        # Calculate loss
        loss = model.get_loss(X, Yoh_)
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch {i}, Loss: {loss.item()}')

def eval(model, X):
    """Arguments:
    - model: type: PTLogreg
    - X: actual datapoints [NxD], type: np.array
    Returns: predicted class probabilites [NxC], type: np.array
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model.forward(X_tensor).detach().numpy()
    return probs
    
if __name__ == "__main__":
    # Inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    torch.manual_seed(100)

    X, Y_ = sample_gmm_2d(6, 2, 100)

    # Pretvori oznake u one-hot notaciju koristeći funkciju iz data.py
    Yoh_ = class_to_onehot(Y_)

    # Pretvori u torch.Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Yoh_tensor = torch.tensor(Yoh_, dtype=torch.float32)

    # Definiraj model
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # Nauči parametre
    train(ptlr, X_tensor, Yoh_tensor, param_niter=1000, param_delta=0.1)

    # Dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)

    # Ispiši performansu
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == Y_)
    print(f'Accuracy: {accuracy}')

    graph_surface(lambda x: eval(ptlr, x)[:, 1], (X.min(axis=0), X.max(axis=0)))
    graph_data(X, Y_, preds)
    plt.show()



