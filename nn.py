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
import numpy as np

from tutioral.v3 import biases

inputs = [1,2,3,2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
In numpy the first elemnet you pass is how the return is index, hense why you use weights first
weights dim (3,4) and inputs dim (4,)
3 is the out and 4 is the same for columns. inputs is also transposed
'''

output = np.dot(weights, inputs) + biases
print(output)