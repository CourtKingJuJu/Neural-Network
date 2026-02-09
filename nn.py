import sys
import matplotlib
import numpy as np

inputs = [1.2, 5.2, 2.1] #output from previous layers 
weights = [3.1, 2.1, 8.7]
bias = 3

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

"""
- One bias per neuron. and one weight for each connection (line from previous layer)
What are Inputs (first layer)
- Could be a vector of values
- What ever vales you have that are tracking (dog, cat, fish)
"""

# Video 2

# Hidden layer of 4 neruons --> Output layer of 3 neruons 

inputs = [1.2, 5.2, 2.1, 1.1]
# 3 sets of 4 weights  for fully connection 3 neurons to the 4 hidden layer neurons 
# 3 bias one for each output neuron
weights1 = [0.2, 0.8, -0.5, 1.5]
weights2 = [3.1, 2.1, 8.7, 1.4]
weights3 = [-0.26, 2.1, 0.17, 1.3]
bias1 = 2
bias2 = 3
bias3 = 0.5

# each line represents one neuron, has its set of weights for each input and its own bias.

output2 = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[2]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[2]*weights3[3] + bias3 ]
        
print(output2)
"""
Entire principal of deep learning is tuening these weights, using backpropgation and different stratgies.
"""

# Video3 