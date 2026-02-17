'''
There is an issue when you get softmax_outputs and put it in log if the cofidence for the right answer is 0 than the log is infinite
then you can't do infinite with some operations. So this is why we actually clip a non important value
Clip by 1-e7

To calculate accuracy you take the actually predictions from your model. Using argmax taking a scalar or one hto vecotr for each row
then compare it tot the y_pred which are scalar or one hot vectors. 
Take a 1 for true 0 for false then mean it and thats the correct prediction % 

Even tho predictions is more true loss is more important as it's how wrong a network is 
'''
# Just adding the loss function to the main code

import numpy as np

# Np arrays are lit
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(softmax_outputs[[0, 1, 2], class_targets])