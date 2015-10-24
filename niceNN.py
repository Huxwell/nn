from itertools import izip
import os
import random
import numpy as np
from PIL import Image as Im

rand = lambda: random.random() - 0.5  # from -0.5 to 0.5)
sigmoid = lambda (x): 1.0 / (1.0 + np.exp(-x))
grad_sigmoid = lambda (x): sigmoid(x) * (1 - sigmoid(x))
signum = lambda (x): 1 if x >= 0 else -1


class Neuron(object):

    signals = []
    sum = 0

    def __init__(self, no_outs, last=False):
        self.weights = rand() if(last) else [rand() for n in xrange(no_outs)]
    # every neutron get no_output synapses,  hidden layers has same no of neurons that output layer


class NN(object):
    old_neurons = []
    neurons = []
    weights = []

    def __init__(self, no_ins, no_outs, no_hidd, learning):
        self.learning = learning
        self.no_outs = no_outs

        self.neurons.append([Neuron(no_outs) for n in xrange(no_ins)])  # input layer
        for l in xrange(no_hidd):
            self.neurons.append([Neuron(no_outs) for n in xrange(no_outs)])
        self.neurons.append([Neuron(no_outs, True) for n in xrange(no_outs)])  # output layer

    def clean_neurons(self):
        for l in self.neurons:
            for n in l:
                n.sum = 0
                n.signals = [0 for i in xrange(self.no_outs)] if l != self.neurons[-1] else 0

    def forward(self, dig_data):
        self.clean_neurons()
        # put a digit in input layer
        for n, d in izip(self.neurons[0], dig_data):
            n.sum = d

        for l_id, l in enumerate(self.neurons):
            for n_id, n in enumerate(self.neurons[l_id]):
                if l_id == len(self.neurons) - 1:
                    print "n.weights: " + str(n.weights)
                    print "n.sum: " + str(n.sum)
                    print "sum * weights: " + str(n.sum * n.weights)
                    print "signals: " + str(self.neurons[l_id][n_id].signals)
                    self.neurons[l_id][n_id].signals = n.sum * n.weights
                    print self.neurons[l_id][n_id].signals
                else:
                    for o in xrange(self.no_outs):
                        self.neurons[l_id][n_id].signals[o] = n.sum * n.weights[o]
                        self.neurons[l_id+1][o].sum += self.neurons[l_id][n_id].signals[o]

    def epoch(self, dig_data, dig_no):
        # self.outs =
        self.forward(dig_data)
        print "sums: " + str([n.sum for n in self.neurons[-1]])
        outs = [n.signals for n in self.neurons[-1]]
        print "outs: " + str(outs)
        proper_outs = [1 if i==dig_no else -1 for i in xrange(10)]
        print "proper outs: " + str(proper_outs)
        errors = [ (proper_out - out) /2 for out, proper_out in izip(outs, proper_outs)]
        print "errors: " + str(errors)
        self.update_weights(errors)
    def update_weights(self, errors):
        print "output layer weights:" + str([n.weights for n in self.neurons[-1]])
        for n_idx, n in enumerate(self.neurons[-1]):
            self.neurons[-1][n_idx].weights = self.neurons[-1][n_idx].weights + self.neurons[-1][n_idx].signals * errors[n_idx] * self.learning

        #for l_idx, l in enumerate(reversed(self.neurons)):

        #self.weights[-1] = [[w[0] + o * e * self.learning] for w, o, e in izip(self.weights[-1],self.outs, errors)]  # last layer
        print "output layer weights:" + str([n.weights for n in self.neurons[-1]])

def ldigits():
    return [[int(p[0] + p[1] + p[2] <= 127 * 3) for p in list(Im.open("data/" + f).getdata())] for f in
            os.listdir("data") if ".bmp" in f]
