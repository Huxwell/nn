from mynn import *
from cnn import *
nn = NN(77, 10, 2, 0.2)
digits = ldigits();
nn.epoch(digits[8], 8)
print nn.neurons[3]

# print NeuralNetwork(77,10,10,2)
