# Hidden layer of 4 neruons --> Output layer of 3 neruons 

inputs = [1, 2, 3, 2.5]
# 3 sets of 4 weights  for fully connection 3 neurons to the 4 hidden layer neurons 


weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
Weights an biases are tuned by the machine, (learnable parameters)
Biases are just offsets, can think of it as b in mx + b. moving where the line starts.
(Wx + b)
'''

layer_outputs = []
for neuron_weights, neuron_biases in zip(weights, biases):
    nn_output = 0
    for n_input, weight in zip(neuron_weights, inputs):
        nn_output += n_input * weight
    nn_output += neuron_biases
    layer_outputs.append(nn_output)

print(layer_outputs)
        
'''
Talking about shape
Array [1, 2, 3, 4]
1D array, Vector, Shape 4 

Array: [1,5,7,2][3,2,1,3]
Shape: (2, 4)
2D Array, Matrix 

This shows us how arrays need to have the same amount of elements in each row

Tensors (Simply): Is an object that can be represented as an Array. 
In deep learning tensors are rpesented as an array

Dotproduct. The reason the wight matrix has the same column number (n) as the input
vector is because this is needed for the dot product to work
'''