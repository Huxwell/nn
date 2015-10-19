# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# im=plt.imread("data/5.bmp")
# np.array(filter(lambda x: x >= 127, im[0][0]))
# print im[0]
# plt.imshow(im)
# plt.show()
# print [p[0]+p[1]+p[2] <= 127 *3 for p in list(Im.open("data/" + [f for f in os.listdir("data") if ".bmp" in f][1]).getdata())]
import os
import random
import numpy as np
from PIL import Image as Im

rand = lambda: random.random() - 0.5 # from -0.5 to 0.5)
sigmoid = lambda(x): 1.0/(1.0+np.exp(-x))
grad_sigmoid = lambda (x): sigmoid(x)*(1-sigmoid(x))

class NN(object):
    neurons=[]
    weights=[]
    # hidden layers has no_outs neutrons each

    def __init__(self, no_ins, no_outs, no_hidd):
        self.neurons.append([0 for l in xrange(no_ins)])
        for l in xrange(1, no_hidd+2):
            self.neurons.append([0 for l in xrange(10)])

        self.weights.append([[rand() for sec in xrange(no_outs)] for fir in xrange(no_ins)])
        for l in xrange(1, no_hidd+1):
            self.weights.append([[rand() for sec in xrange(no_outs)] for fir in xrange(no_outs)])
        self.weights.append([[rand()] for fir in xrange(no_outs)])

    def forward(self,dig_data,dig_no):
            self.neurons[0] = dig_data
            for layerId, layer in enumerate(self.neurons):
                if layerId == self.neurons.__len__() - 1:
                    return
                for idx, i in enumerate(self.neurons[layerId]):
                    for idx2, n in enumerate(self.neurons[layerId+1]):
                        self.neurons[layerId+1][idx2] += i * sigmoid(self.weights[layerId][idx][idx2])




def ldigits():
    return [[int(p[0]+p[1]+p[2] <= 127 * 3) for p in list(Im.open("data/" + f).getdata())] for f in os.listdir("data") if ".bmp" in f]




def forward(neurons, weights, dig_data, dig_no):
    neurons[0] = dig_data
    for layerId, layer in enumerate(neurons):
        if layerId == neurons.__len__() - 1:
            return neurons
        for idx, i in enumerate(neurons[layerId]):
            for idx2, n in enumerate(neurons[layerId+1]):
                neurons[layerId+1][idx2] += i * sigmoid(weights[layerId][idx][idx2])

        # for idx, i in enumerate(neurons[0]):
        #     for idx2, n in enumerate(neurons[1]):
        #         neurons[1][idx2] += i* weights[0][idx][idx2]
        # for idx, i in enumerate(neurons[1]):
        #     for idx2, n in enumerate(neurons[2]):
        #         neurons[2][idx2] += i* weights[1][idx][idx2]
        # for idx, i in enumerate(neurons[2]):
        #     for idx2, n in enumerate(neurons[3]):
        #         neurons[3][idx2] += i* weights[2][idx][idx2]
    return neurons


nn = NN(77,10,2)
print nn.weights[3][9]

digits = ldigits();
nn.forward(digits[8], 8)

print nn.neurons[3]
