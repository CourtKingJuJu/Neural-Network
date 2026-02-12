#  Video 1

"""
- One bias per neuron. and one weight for each connection (line from previous layer)
What are Inputs (first layer)
- Could be a vector of values
- What ever vales you have that are tracking (dog, cat, fish)
"""

# Video 2

"""
Entire principal of deep learning is tuening these weights, using backpropgation and different stratgies.
"""

# Video3 

'''
In numpy the first elemnet you pass is how the return is index, hense why you use weights first
weights dim (3,4) and inputs dim (4,)
3 is the out and 4 is the same for columns. inputs is also transposed
'''

# Video4
""""
We use batches when fitting a line to help generalization. THe batch size ish ow many samples it sees at a time
and then it updates the line accordingly, this helps generalization and reduces over fitting.
"""

"""
we get to see how the weight matrix is classes in sample (in this case 4) by the amount of neurons in the layer. thi is XW + b form
we sometimes see the Wx Form where it might be written as 5 x 4. 
"""


import numpy as np
import nnfs 
import matplotlib as plt
from nnfs.datasets import spiral_data

# nnfs.init()
class LayerDense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    

np.random.seed(0)

X, y = spiral_data(100, 3)


layer1 = LayerDense(2, 5)
activation = ActivationReLU()
layer1.forward(X)
activation.forward(layer1.output)

print(activation.output)

