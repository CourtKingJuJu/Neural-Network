# Example layers 
import numpy as np

# Batched input
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13],]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87],]

biases2 = [2, 3, 0.5]


layer1_outputs = np.dot(weights2, np.matrix(inputs).transpose()) + biases

layer2_outputs = np.dot(weights, np.array(layer1_outputs).transpose() + biases2)

print(layer2_outputs)

np.random.seed(0)

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]


class Layer_dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_dense(4, 5)
layer2 = Layer_dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)