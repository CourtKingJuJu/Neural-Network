# Neural-Network

Creating a neural network from scratch
Following Sentdex NN from scratch [tutorial series](https://www.youtube.com/watch?v=lGLto9Xd7bU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=2)

## Learning points

- Most learning points are in Tutioral Videos
- Each neuron has it's own set of weights for the previous input, with the only exception being the input layer. They also have their own bais and output
- We can't change the output of neurons in a crazy way using square root or abs value because we need to back propgate
- Exponetiation and normalize is what makes up softmax or the formula Si,j = e^zi,j / Su e^zi,j
- Something people with do to input before exponetiation (softmax) is subtracting max value from everything.
- This makes our range between 0,1 because 0 in exp is 1 and everything else will be between that
