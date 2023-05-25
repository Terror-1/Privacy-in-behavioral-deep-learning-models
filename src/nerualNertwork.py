import numpy as np 
np.random.seed(0)

X = [[1,0,1,0],[1,0,1,1],[0,1,0,1]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
Layer1 = Layer_Dense(4,5)
Layer2 = Layer_Dense(5,2)
Layer1.forward(X)
print(Layer1.output)
Layer2.forward(Layer1.output)
print(Layer2.output)
