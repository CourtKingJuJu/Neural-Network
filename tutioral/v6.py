import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
 

print(exp_values)

# Need to normalize this output
# Take the sume of all the neuros then divide each exp value by the norm base
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))

import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
 

print(exp_values)

# Need to normalize this output
# Take the sume of all the neuros then divide each exp value by the norm base
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))

class ActivationSoftMax:
    def forward(self, inputs):
        # if the input is in batch form this could be bad because it takes the highest input in every batch
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdim=True)
        self.output = probabilites
