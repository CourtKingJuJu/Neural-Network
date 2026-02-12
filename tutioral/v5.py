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
outputs = []
# Created the ReLU activation function 
for i in inputs:
    output.append(max(0, i))
