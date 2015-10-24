#from niceNN import *
#from cnn import *
#nn = NN(77, 10, 2, 0.2)
#digits = ldigits();
#nn.epoch(digits[8], 8)
# print NeuralNetwork(77,10,10,2)
from singleh import *
nn = NN();

print('\n'.join([''.join([' {:4}'.format(item) for item in row])
      for row in nn.ws_o]))

digs = ldigits()
print len(nn.ws_h[0])
nn.forward(digs[8],8)
