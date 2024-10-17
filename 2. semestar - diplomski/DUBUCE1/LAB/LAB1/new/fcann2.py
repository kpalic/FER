import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, graph_surface, graph_data, class_to_onehot

def softmax(inputLogits):

    logits = inputLogits - np.max(inputLogits, axis=1, keepdims=True)  # Numerička stabilnost
    exp_logits = np.exp(logits)
    exp_logits = np.clip(exp_logits, 1e-12, np.inf)  # Izbjegavanje prenaprezanja i underflow-a
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax_probs

class FCANN2():
    def __init__(self, inputSize, hiddenSize, outputSize, learning_rate=0.05, reg_lambda=1e-3):
        self.W_hidden = np.random.randn(inputSize, hiddenSize) * 10
        self.b_hidden = np.zeros(hiddenSize) * 10

        self.W_output = np.random.randn(hiddenSize, outputSize) *10 
        self.b_output = np.zeros(outputSize) * 10 

        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def forward(self, input):
        self.input = input

        # propagacija unaprijed
        self.hidden_output = np.dot(input, self.W_hidden) + self.b_hidden
        self.hidden_output = np.maximum(0, self.hidden_output)  # ReLU

        self.final_output = np.dot(self.hidden_output, self.W_output) + self.b_output
        self.final_output = softmax(self.final_output)  # softmax

        return self.final_output

    def compute_loss(self, true_output):
        epsilon = 1e-12  # mali broj za numeričku stabilnost
        log_probs = -np.log(self.final_output + epsilon)  # izbjegavanje log(0) u računanju gubitka
        loss = np.sum(true_output * log_probs) / true_output.shape[0]
        # regularizacija (L2)
        reg_loss = self.reg_lambda / 2 * (np.sum(np.square(self.W_hidden)) + np.sum(np.square(self.W_output)))
        loss += reg_loss
        return loss

    def backward(self, true_output):
        # Backward propagacija
        output_error = self.final_output - true_output  # derivacija gubitka po izlazu
        output_delta = output_error  # za softmax s NLL

        # Derivacija gubitka po W2 i b2
        grad_W_output = np.dot(self.hidden_output.T, output_delta) + self.reg_lambda * self.W_output
        grad_b_output = np.sum(output_delta, axis=0)


        # Derivacija gubitka po h1
        hidden_error = np.dot(output_delta, self.W_output.T)
        
        # Derivacija gubitka po s1
        hidden_delta = hidden_error * (self.hidden_output > 0)  # ReLU derivacija

        # Derivacija gubitka po W1 i b1
        grad_W_hidden = np.dot(self.input.T, hidden_delta) + self.reg_lambda * self.W_hidden
        grad_b_hidden = np.sum(hidden_delta, axis=0)

        self.W_output -= self.learning_rate * grad_W_output
        self.b_output -= self.learning_rate * grad_b_output
        self.W_hidden -= self.learning_rate * grad_W_hidden
        self.b_hidden -= self.learning_rate * grad_b_hidden

    def train(self, input, output, n_iter=100000):
        for epoch in range(int(n_iter)):
            self.forward(input)
            loss = self.compute_loss(output)
            self.backward(output)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def classify(self, input):
        self.forward(input)
        return np.argmax(self.final_output, axis=1)

if __name__ == "__main__":
    np.random.seed(100)

    K = 6  # Broj komponenti
    C = 3  # Broj klasa
    N = 10  # Broj uzoraka po komponenti
    
    # Generiraj podatke
    X, Y_ = sample_gmm_2d(K, C, N)  # Generiraj podatke s 6 komponenti, 2 klase, 10 uzoraka po komponenti
    Y = class_to_onehot(Y_)          # Pretvori klase u one-hot kodiranje

    # Treniraj model
    model = FCANN2(inputSize=2, hiddenSize=5, outputSize=C, learning_rate=0.05, reg_lambda=1e-3)
    model.train(X, Y, n_iter=100000)

    # Klasifikacija podataka
    Y_pred = model.classify(X)

    # Grafički prikaz površine odluke
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    def decision_function(X):
        return model.forward(X)[:, 1]  # Vraćamo vjerojatnosti za jednu klasu

    graph_surface(decision_function, rect, offset=0)
    
    # Grafički prikaz podataka
    graph_data(X, Y_, Y_pred)

    plt.show()
