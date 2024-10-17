import torch
import torch.nn as nn
import torch.optim as optim

# Funkcija za računanje gradijenata analitički
def analytic_gradients(X, Y, a, b):
    N = X.size(0)
    Y_ = a * X + b
    diff = Y - Y_
    grad_a = -2 * torch.sum(X * diff) / N
    grad_b = -2 * torch.sum(diff) / N
    return grad_a, grad_b

# Definicija računskog grafa
# Podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Proizvoljan broj točaka
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
Y = torch.tensor([3.0, 5.0, 7.0, 9.0])

# Optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # Afin regresijski model
    Y_ = a * X + b

    diff = (Y - Y_)

    # Kvadratni gubitak
    loss = torch.sum(diff**2) / X.size(0)

    # Računanje gradijenata
    loss.backward()

    # Računanje analitičkih gradijenata
    grad_a, grad_b = analytic_gradients(X, Y, a, b)

    # Ispis rezultata prije koraka optimizacije
    print(f'step: {i}, loss: {loss.item()}, Y_: {Y_.detach().numpy()}, a: {a.item()}, b: {b.item()}')
    print(f'grad_a (autograd): {a.grad.item()}, grad_b (autograd): {b.grad.item()}')
    print(f'grad_a (analytic): {grad_a.item()}, grad_b (analytic): {grad_b.item()}')

    # Korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()
