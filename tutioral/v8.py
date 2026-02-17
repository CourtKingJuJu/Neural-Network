'''
There is an issue when you get softmax_outputs and put it in log if the cofidence for the right answer is 0 than the log is infinite
then you can't do infinite with some operations. So this is why we actually clip a non important value
Clip by 1-e7
'''
# Just adding the loss function to the main code

import numpy as np

# Np arrays are lit
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(softmax_outputs[[0, 1, 2], class_targets])