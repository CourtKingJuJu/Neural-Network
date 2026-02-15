# Activiation Functions

'''
You can think of activation functions as happening on the backside of the node
Different types
- Step function (0 or 1)
- Sigmoid: Step climb kidnda
- ReLU: O or max 

Without activation functions NN's are linear and therefore can't fit certain things like XOR or sin waves
These functions are not linear, they are almost linear but have little things that make them quick. 
ReLU deep dive, allows you to set kinda of min max bounds between neuron pairs. This helps fit different things
This kinda shows feature, like for example in the sin wave you might have one neuron pair that bounds the slope of the first
part of the up slope. then the second pair represents the peak, and then the down slope etc..

In other works it one neuron set could set an activation point and a deactivation point. 
'''
inputs = []
output = []
# Created the ReLU activation function 
for i in inputs:
    output.append(max(0, i))


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
