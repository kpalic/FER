import numpy as np


def sigmoid(x):
    result = 0
    for i in x:
        sigm = 1 / 1 + np.exp(-i)
        result = result + sigm
    return result

class FCANN2():
# fully connected artificial neural network with 2 layers using numpy
    
    def __init__(self, inputSize, hiddenSize, outputSize):

        self.W_hidden = np.random.randn(inputSize, hiddenSize)
        self.b_hidden = np.random.randn(hiddenSize)

        self.W_output = np.random.randn(hiddenSize, outputSize)
        self.b_output = np.random.randn(outputSize)
    

    
    def forward(self, input):
        # propagacija unaprijed
        hiddenOutput = np.dot(input, self.W_hidden) + self.b_hidden
        hiddenOutput = max(0, hiddenOutput) # RELU - zglobnica
        final_output = np.dot(hiddenOutput, self.W_output) + self.b_output
        final_output = sigmoid(final_output)
        return final_output


    def train(input, output):
    

    def clasify(input):