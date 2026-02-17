# Calculating loss

'''
Loss is something that needs to be caculated. it is some value of the distrbuition of the confidence score of the model output
there are different loss functions 
1. Mean Absolute Error
- Closer to the correct value the mean gets closer to 0 and the further the bigger

The loser functio nof choice for classification and softmax is 
Categorical cross entropy. 
Is pretty much the -log(of the classifcation function)

One hot encoding
vector where one vector is 1 and the other are 0. One classification I.E. [0, 1, 0] where: 3 classes, 1 label 
This basicall sums up to -(one hot vector spot (1) x log(predicted class value from ome hot))

Assuming natural log ln = base e (eular's number)
log is solving for x in the equation e**x = b. 


import numpy as np
import math
b = 5.2

print(np.log(b))
print(math.e ** (np.log(b)))
'''

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0] # 0.7 is one hot
 # this is loss calculated using categorical cross entropy
loss = -(math.log(softmax_output[0]) * target_output[0]) # should be + all over0 spots

print(loss)
# Loss is lower where cofindence in correct class is higher
print(-math.log(0.7))
print(-math.log(0.5))