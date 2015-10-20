import os
import random
import numpy as np
from PIL import Image as Im

rand = lambda: random.random() - 0.5  # from -0.5 to 0.5)
sigmoid = lambda (x): 1.0 / (1.0 + np.exp(-x))
grad_sigmoid = lambda (x): sigmoid(x) * (1 - sigmoid(x))
signum = lambda (x): 1 if x >= 0 else -1


class NN(object):
    neurons = []
    weights = []
    # hidden layers has no_outs neutrons each

    def __init__(self, no_ins, no_outs, no_hidd):
        self.neurons.append([0 for l in xrange(no_ins)])
        for l in xrange(1, no_hidd + 2):
            self.neurons.append([0 for l in xrange(10)])

        self.weights.append([[rand() for sec in xrange(no_outs)] for fir in xrange(no_ins)])
        for l in xrange(1, no_hidd + 1):
            self.weights.append([[rand() for sec in xrange(no_outs)] for fir in xrange(no_outs)])
        self.weights.append([[rand()] for fir in xrange(no_outs)])

    def clean_neurons(self):
        for l_id, l in enumerate(self.neurons):
            for n_id, n in enumerate(self.neurons[l_id]):
                self.neurons[l_id][n_id] = 0


    def forward(self, dig_data, dig_no):
        self.clean_neurons()
        # put a digit in input layer
        self.neurons[0] = dig_data[:]
        for l_id, layer in enumerate(self.neurons):
            # if l_id = 4 - 1 = 3  so if we are in output layer
            if l_id == self.neurons.__len__() - 1:
                return [signum(no * self.weights[l_id][no][0]) for no in self.neurons[3]]
            for idx, i in enumerate(self.neurons[l_id]):
                for idx2, n in enumerate(self.neurons[l_id + 1]):
                    self.neurons[l_id + 1][idx2] += signum(i * self.weights[l_id][idx][idx2])


def ldigits():
    return [[int(p[0] + p[1] + p[2] <= 127 * 3) for p in list(Im.open("data/" + f).getdata())] for f in
            os.listdir("data") if ".bmp" in f]


nn = NN(77, 10, 2)

digits = ldigits();
nn.forward(digits[8], 8)
#nn.forward(digits[8], 8)


print nn.forward(digits[8], 8)

#print nn.neurons[3]

